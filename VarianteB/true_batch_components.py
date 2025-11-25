import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from data_utils import HybridEndToEndDataset as BaseDataset

# DATASET WRAPPER 
class BatchableHybridDataset(BaseDataset):
    """
    Wrapper que extiende el Dataset base para hacerlo compatible con el DataLoader 
    de PyTorch Geometric. Encapsula los datos heterogéneos (Serie Temporal + Grafo) 
    en objetos 'Data' que soportan 'collate' automático (agrupamiento inteligente).
    """
    def __getitem__(self, idx):
        # Recuperamos la tupla de datos crudos del dataset padre
        # (x_lstm: secuencia temporal, x_gnn: features de nodos, etc.)
        raw_data = super().__getitem__(idx)
        x_lstm, x_gnn, edge_index, edge_weight, y_target, _ = raw_data

        # Garantizamos que el objetivo sea un tensor flotante (requisito para MSELoss)
        if not isinstance(y_target, torch.Tensor):
            y_target = torch.tensor(y_target, dtype=torch.float32)

        # Construcción del objeto Data de PyG
        # Este objeto agrupa la topología del grafo y la secuencia temporal.
        # PyG se encargará de fusionar 'x', 'edge_index' y 'edge_attr' en un supergrafo
        # cuando creemos un batch. 'x_lstm' se apilará como un tensor normal.
        data = Data(
            x=x_gnn,                      # Features de los nodos (N_nodes, N_features)
            edge_index=edge_index.long(), # Lista de adyacencia (2, N_edges)
            edge_attr=edge_weight,        # Pesos de las aristas (Correlación)
            y=y_target.view(1, -1),       # Etiqueta target (Reshaped para consistencia)
            x_lstm=x_lstm.unsqueeze(0)    # Secuencia LSTM (1, Seq_Len, Features)
        )
        
        # Guardamos el índice local del nodo objetivo. 
        # Esto es vital para saber qué nodo mirar dentro del grafo gigante del batch.
        data.target_idx = torch.tensor([self.target_idx], dtype=torch.long)
        return data

# ARQUITECTURA DEL MODELO HIBRIDO

class TrueBatchHybridModel(nn.Module):
    """
    Implementación de la arquitectura híbrida LSTM-GNN.
    Procesa simultáneamente:
    1. Dinámica Temporal (LSTM): Historia de precios del activo individual.
    2. Dinámica Espacial (GNN): Relaciones de mercado con otros activos.
    """
    def __init__(self, lstm_hidden, lstm_layers, gnn_hidden, gnn_out, mlp_hidden_dims, dropout, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes # Total de activos en el universo (ej. 10)
        
        # RAMA TEMPORAL 
        # Captura dependencias secuenciales (momentum, reversión a la media).
        self.lstm = nn.LSTM(
            input_size=1,             # Entrada: 1 feature (Precio normalizado)
            hidden_size=lstm_hidden,  # Dimensión del vector de memoria
            num_layers=lstm_layers,
            batch_first=True, 
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.lstm_drop = nn.Dropout(dropout)

        # RAMA RELACIONAL (Graph-Based)
        # Captura la influencia de activos correlacionados
        # Usamos GCN (Graph Convolutional Network) para propagar información entre vecinos.
        self.conv1 = GCNConv(1, gnn_hidden)       # Capa 1: Input -> Latente
        self.conv2 = GCNConv(gnn_hidden, gnn_out) # Capa 2: Latente -> Embedding Relacional
        self.gnn_drop = nn.Dropout(dropout)
        self.act = nn.ReLU() # Función de activación no lineal

        # FUSIÓN E INFERENCIA 
        # El vector final combina lo que "recuerda" el LSTM y lo que "ve" el GNN.
        in_dim = lstm_hidden + gnn_out
        
        layers = []
        prev = in_dim
        # Construcción dinámica del MLP (Perceptrón Multicapa) final
        for h in mlp_hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        
        layers.append(nn.Linear(prev, 1)) # Salida escalar (Predicción de precio)
        self.mlp = nn.Sequential(*layers)

    def forward(self, data):
        """
        Paso Forward. Maneja explícitamente el 'Disjoint Union Batching' de PyG.
        """
        
        # 1. PROCESAMIENTO TEMPORAL (LSTM Branch)
        x_seq = data.x_lstm
        # Ajuste dimensional: (Batch, Seq, 1)
        if x_seq.dim() == 4: x_seq = x_seq.squeeze(1)
             
        # out_lstm contiene todos los pasos, h_n es el estado final.
        out_lstm, (h_n, _) = self.lstm(x_seq)
        
        # Tomamos el último estado oculto como el "Temporal Embedding"
        # Representa el resumen de toda la ventana histórica.
        h_lstm = h_n[-1] 
        h_lstm = self.lstm_drop(h_lstm)

        # 2. PROCESAMIENTO RELACIONAL (GNN Branch)
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        # Propagación de mensajes (Message Passing)
        x = self.conv1(x, edge_index, edge_weight) # Convolución 1
        x = self.act(x)
        x = self.gnn_drop(x)
        x = self.conv2(x, edge_index, edge_weight) # Convolución 2 (Salida: Relational Embedding)
        x = self.act(x)

        # 3. RECUPERACIÓN DE NODOS (Target Retrieval) - CRÍTICO
        # En PyG, un batch de B grafos con N nodos se convierte en un solo Grafo Gigante de B*N nodos.
        # Consecuencia: Los índices de los nodos se desplazan linealmente.
        # Si cada grafo tiene 10 nodos:
        # El nodo objetivo del Grafo 0 está en el índice global: 0
        
        batch_size = data.num_graphs
     
        # 'offsets' calcula este desplazamiento para recuperar el embedding correcto de cada grafo.
        offsets = torch.arange(batch_size, device=x.device) * self.num_nodes
        
        # Calculamos los índices globales en el grafo gigante
        target_indices = offsets + data.target_idx
        
        # Extraemos SOLO los embeddings de los activos que queremos predecir
        h_gnn = x[target_indices]

        # 4. FUSIÓN HÍBRIDA
        # Concatenamos el contexto temporal y el contexto relacional.
        combined = torch.cat([h_lstm, h_gnn], dim=1) 
        
        # 5. PREDICCIÓN FINAL
        out = self.mlp(combined)
        return out.squeeze(-1) # Retornamos forma (Batch_Size,)