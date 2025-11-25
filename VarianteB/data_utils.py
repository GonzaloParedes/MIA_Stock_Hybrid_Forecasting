# data_utils.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List
from train_gnn_fixed import build_graph_and_features

@dataclass
class TrainConfig:
    lstm_hidden: int = 64
    lstm_layers: int = 2
    gnn_hidden: int = 64
    gnn_out: int = 32
    mlp_hidden_dims: tuple = (64, )
    dropout: float = 0.5   
    lr: float = 0.001       
    epochs: int = 40
    seed: int = 42
    batch_size: int = 21
    patience: int = 8
    
    # GNN Simplificada
    pearson_corr: float = 0.5  # Bajé el umbral a 0.5 (te dio mejor resultado)

def build_windows_from_series(series_values: np.ndarray, dates: pd.Index, seq_len: int):
    vals = series_values.astype("float32")
    X_list, y_list, d_list = [], [], []
    for t in range(seq_len - 1, len(vals) - 1):
        # Ventana de historia: t - seq_len + 1 hasta t
        past = vals[t - (seq_len - 1): t + 1]
        # Objetivo: Precio en t+1
        target = vals[t + 1] 
        # Reshape para (Seq, Features=1)
        X_list.append(past.reshape(-1, 1))
        y_list.append(target)
        d_list.append(dates[t + 1])
    if not X_list: return np.empty((0, seq_len, 1)), np.empty((0,)), []
    return np.stack(X_list), np.array(y_list), d_list

class HybridEndToEndDataset(Dataset):
    def __init__(self, prices, tickers, target_ticker, seq_len, W, 
                 train_end_date, horizon_days, feature_mode, adj_mode, 
                 price_min, price_max, cfg: TrainConfig, split="train"):
        
        self.target_idx = tickers.index(target_ticker)
        
        # Normalización
        # Se aplican los valores min/max calculados en el conjunto de entrenamiento
        # para evitar 'Data Leakage' (fuga de información) del conjunto de test.
        series_target = prices[target_ticker].values.astype("float32")
        p_min_val = float(price_min[self.target_idx])
        p_max_val = float(price_max[self.target_idx])
        denom = p_max_val - p_min_val
        if abs(denom) < 1e-6: denom = 1.0

        # Serie normalizada lista para ser procesada
        series_norm_values = (series_target - p_min_val) / denom
        
        # Generación de Secuencias LSTM 
        # Aca se crean las ventanas de tiempo
        X_lstm_all, _, dates_lstm = build_windows_from_series(series_norm_values, prices.index, seq_len)

        # Mapa hash para acceso rápido: Fecha -> Índice de secuencia LSTM
        self.date_to_lstm_idx = {d: i for i, d in enumerate(pd.to_datetime(dates_lstm))}
        self.dates = prices.index
        
        # Definición de Índices Válidos 
        # Necesitamos tener al menos 'W' días de historia para calcular la correlación inicial.
        graph_indices = list(range(seq_len - 1, len(self.dates) - 1))
        train_idxs, test_idxs = [], []
        train_end_date = pd.to_datetime(train_end_date)
        
        # Separación temporal estricta (Cronológica)
        for idx in graph_indices:
            next_day = self.dates[idx + 1]
            if next_day not in self.date_to_lstm_idx: continue
            if next_day <= train_end_date:
                train_idxs.append(idx)
            else:
                test_idxs.append(idx)

        # Selección de índices según si estamos entrenando o testeando
        target_indices = train_idxs if split == "train" else test_idxs[:horizon_days]
        
        self.cache = []
        # Construcción del grafo en CPU para no saturar VRAM
        device_cpu = torch.device("cpu") 
        

        #  Bucle Principal de Construcción de Muestras 
        for idx in target_indices:
            # Día actual (t)
            end_day = self.dates[idx]
            # Día objetivo a predecir (t+1)
            next_day = self.dates[idx + 1]
            
            # Recuperar Input LSTM
            lstm_idx = self.date_to_lstm_idx[next_day]
            x_lstm = torch.from_numpy(X_lstm_all[lstm_idx]).float()

            # Construir Grafo Dinámico (GNN Input)
            # 'build_graph_and_features' toma los precios desde (t - W) hasta t
            # y calcula la matriz de correlación de Pearson sobre esa ventana.
            X_nodes, edge_index, edge_weight, y_all = build_graph_and_features(
                prices=prices, 
                tickers=tickers, 
                end_day=end_day, 
                next_day=next_day,
                pearson_min_abs_corr=cfg.pearson_corr, # Solo pasamos esto
                device=device_cpu,
                price_min=price_min, 
                price_max=price_max
            )
            # Preparar Target
            if y_all.dim() > 1: y_all = y_all.view(-1)
            y_target = y_all[self.target_idx].float()

            # Cacheamos la tupla completa: (LSTM Input, GNN Input, Target)
            self.cache.append((x_lstm, X_nodes, edge_index, edge_weight, y_target, next_day))

    def __len__(self): return len(self.cache)
    def __getitem__(self, idx): return self.cache[idx]