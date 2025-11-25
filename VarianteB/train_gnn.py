import numpy as np
import pandas as pd
import torch
from typing import List

#UTILIDADES INTERNAS

def dense_adj_to_edge_index(adj_matrix):
    if not isinstance(adj_matrix, torch.Tensor):
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
    indices = torch.nonzero(adj_matrix, as_tuple=False).t()
    weights = adj_matrix[indices[0], indices[1]]
    return indices, weights

def maybe_add_self_loops(adj_matrix, add_self_loops=True):
    if not add_self_loops: return adj_matrix
    if isinstance(adj_matrix, np.ndarray):
        np.fill_diagonal(adj_matrix, 1.0)
    elif isinstance(adj_matrix, torch.Tensor):
        diag_idx = torch.arange(adj_matrix.size(0))
        adj_matrix[diag_idx, diag_idx] = 1.0
    return adj_matrix

def adj_from_pearson(prices_window, pearson_min_abs_corr=0.5):
    """
    Construye adyacencia ponderada basada en correlación lineal.
    Mantiene la magnitud de la correlación si supera el umbral.
    """
    # Calculamos retornos y la matriz de correlación
    # (Pandas suele devolver float64, lo convertimos a float32 para ahorrar memoria)
    corr_matrix = prices_window.pct_change().corr().fillna(0).values.astype(np.float32)
    
    # MATRIZ PONDERADA:
    # Si |corr| > umbral  --> Ponemos |corr| (Peso real)
    # Si |corr| <= umbral --> Ponemos 0.0    (Desconexión)
    adj = np.where(
        np.abs(corr_matrix) > pearson_min_abs_corr, 
        np.abs(corr_matrix), 
        0.0
    )
    
    return adj

def build_graph_and_features(
    prices: pd.DataFrame,
    tickers: List[str],
    end_day: pd.Timestamp,
    next_day: pd.Timestamp,
    pearson_min_abs_corr: float, 
    device: torch.device,
    price_min: np.ndarray,
    price_max: np.ndarray,
):
    """
    Construye el grafo usando:
    1. Estructura: Expanding Window (Toda la historia hasta hoy).
    2. Features: Global Min-Max Scaling (Usando estadísticas de los 2 años).
    """
    P = prices[tickers].dropna().sort_index()
    pos = P.index.get_loc(end_day)
    
    # 1. GRAFO (ESTRUCTURA) -> EXPANDING WINDOW
    # Usamos toda la historia disponible para calcular la correlación
    win_prices_full_history = P.iloc[: pos + 1]

    A = adj_from_pearson(
        win_prices_full_history,
        pearson_min_abs_corr=pearson_min_abs_corr
    )
    A = maybe_add_self_loops(A, add_self_loops=True)
    edge_index, edge_weight = dense_adj_to_edge_index(A)
    
    # FEATURES -> GLOBAL MIN-MAX SCALING
    # Obtenemos el precio de cierre de hoy (end_day)
    current_prices = P.loc[end_day, tickers].values.astype(np.float32)
    
    # Normalizamos usando los Min/Max globales del periodo de entrenamiento.
    
    denom = price_max - price_min
    denom = np.where(np.abs(denom) < 1e-8, 1.0, denom)
    
    x_norm = (current_prices - price_min) / denom
    
    # Convertimos a tensor (N_nodos, 1)
    X = torch.tensor(x_norm, dtype=torch.float32)[:, None].to(device)

    # TARGET (Etiqueta para el MSE)
    close_next = P.loc[next_day, tickers].values.astype(np.float32)
    close_next_norm = (close_next - price_min) / denom
    y = torch.tensor(close_next_norm, dtype=torch.float32)[:, None].to(device)

    edge_index = edge_index.to(device).long()
    edge_weight = edge_weight.to(device).float()

    return X, edge_index, edge_weight, y