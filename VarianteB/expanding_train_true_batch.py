# expanding_train_true_batch.py
import sys, os, time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from torch_geometric.loader import DataLoader 
from torch.utils.data import Subset           

from data_utils import TrainConfig

from true_batch_components import BatchableHybridDataset, TrueBatchHybridModel

# ==============================================================================
# FUNCIÓN SET_SEED (Integrada aquí para evitar dependencias externas)
# ==============================================================================
def set_seed(seed=42):
    """Fija las semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================================================================
# LÓGICA DE ENTRENAMIENTO EXPANDING WINDOW
# ==============================================================================

def expanding_train_true_batch(
    target_ticker: str,
    tickers: list,
    train_end: str,
    training_years: int,
    horizon_days: int,
    seq_len: int,
    W: int,
    cfg: TrainConfig,
    preloaded_data: pd.DataFrame = None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Fijamos semilla al inicio de cada ticker
    set_seed(cfg.seed)
    
    # Ordenamos los tickers
    tickers = sorted(list(set(tickers)))
    
    
    # Lógica para usar datos precargados
    train_end_date = pd.to_datetime(train_end)
    if preloaded_data is not None:
        missing = [t for t in tickers if t not in preloaded_data.columns]
        if missing: raise ValueError(f"Faltan columnas: {missing}")
        all_prices = preloaded_data[tickers].copy()
    else:
        start_date = train_end_date - relativedelta(years=training_years)
        end_fetch = train_end_date + pd.Timedelta(days=horizon_days * 2 + 50)
        raw = yf.download(tickers, start=start_date, end=end_fetch, auto_adjust=True, progress=False)
        try:
            df_tmp = raw["Close"]
        except KeyError:
            df_tmp = raw
        all_prices = df_tmp.ffill().bfill()[tickers].copy()

    # Manejo de valores nulos residuales
    if all_prices.isnull().values.any():
        all_prices = all_prices.fillna(0.0)

    # Dado que la GNN emite predicciones para todos los nodos
    # necesitamos este índice para enmascarar la salida y calcular el error solo sobre el activo de interés.
    target_idx = tickers.index(target_ticker)

    # Contenedores para almacenar las métricas de evaluación paso a paso.
    # Se almacenan tanto en el espacio latente (normalizado [0,1]) para análisis de convergencia,
    # como en el espacio original ($) para evaluación de impacto financiero.
    preds_norm_list, true_norm_list = [], [] # Espacio (normalizado)
    preds_denorm, truth_denorm, dates_test = [], [], [] # Espacio real
    
    
    current_train_end = train_end_date
    total_start_time = time.time()

    for step in range(horizon_days):
        print(f"STEP {step+1}/{horizon_days} | Fecha Corte: {current_train_end.date()}")

        # Cortar datos
        available_data = all_prices.loc[:current_train_end].copy()
        
        p_min = available_data.min(axis=0).values.astype(np.float32)
        p_max = available_data.max(axis=0).values.astype(np.float32)
        
        if len(available_data) < W + seq_len: 
            print("Datos insuficientes.")
            break

        # GENERACIÓN DEL DATASET
        print("   Generando Dataset y Grafos (Cacheando)... ", end="")
        t0 = time.time()

        # Instanciación del Dataset.
        # Aquí se fusionan las series temporales (LSTM) con la estructura de grafo (GNN).
        full_ds = BatchableHybridDataset(
            prices=available_data, tickers=tickers, target_ticker=target_ticker,
            seq_len=seq_len, W=W, train_end_date=current_train_end, horizon_days=0,
            feature_mode="close_minmax", adj_mode="paper",
            price_min=p_min, price_max=p_max, cfg=cfg, split="train"
        )
        print(f"Listo en {time.time()-t0:.2f}s")
        
        n_samples = len(full_ds)
        if n_samples == 0: break

        # División Train/Validation Cronológica
        # NO usamos shuffle aleatorio global para respetar la temporalidad de la serie.
        indices = np.arange(n_samples) 
        split_point = int(n_samples * 0.8)
        
        ds_train = Subset(full_ds, indices[:split_point])
        ds_val = Subset(full_ds, indices[split_point:])
        
        train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)

        # INICIALIZACIÓN DEL MODELO 
        # Se reinicializa el modelo en cada paso para evitar sobreajuste a la historia lejana
        model = TrueBatchHybridModel(
            lstm_hidden=cfg.lstm_hidden, lstm_layers=cfg.lstm_layers,
            gnn_hidden=cfg.gnn_hidden, gnn_out=cfg.gnn_out,
            mlp_hidden_dims=cfg.mlp_hidden_dims, 
            dropout=cfg.dropout,
            num_nodes=len(tickers)
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        criterion = nn.MSELoss()

        # ENTRENAMIENTO (AJUSTE) 
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(cfg.epochs):
            model.train()
            train_loss = 0
            for batch in train_loader:
                #Mueve los tensores de los datos a gpu
                batch = batch.to(device)
                #Borra los gradientes
                optimizer.zero_grad()
                #Forward Pass: Inferencia del modelo
                pred = model(batch) 
                loss = criterion(pred, batch.y.view(-1))
                #Backpropagation
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                #Aplica los cambios a los pesos de la red neuronal.
                optimizer.step()
                #Te da el error promedio del lote (batch) -  batch.num_graphs te dice cuántos grafos hubo en el batch
                #uma total de errores
                train_loss += loss.item() * batch.num_graphs
            # Calculo de loss promedio por epoca
            avg_train = train_loss / len(ds_train) if len(ds_train) > 0 else 0

            # Evaluación en conjunto de validación
            model.eval()
            val_loss = 0
            if len(ds_val) > 0:
                with torch.no_grad():
                    for batch in val_loader:
                        #Mueve los tensores de los datos a gpu
                        batch = batch.to(device)
                        pred = model(batch)

                        #fuerza a que batch.y se "aplane" (flatten) a una sola dimensión [11]
                        v_loss = criterion(pred, batch.y.view(-1))
                        #suma total de errores
                        val_loss += v_loss.item() * batch.num_graphs
                avg_val = val_loss / len(ds_val)

                # Estrategia de Early Stopping
                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if epoch % 5 == 0 or epoch == 0:
                    print(f"[Ep {epoch+1:02d}] Train: {avg_train:.6f} | Val: {avg_val:.6f} | Best: {best_val_loss:.6f}")

                if patience_counter >= cfg.patience:
                    print(f"Early Stopping en Epoch {epoch+1}")
                    break # Parada temprana por falta de mejora
                
        # Cargar los mejores pesos obtenidos durante el entrenamiento
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # PREDICCIÓN 

        # Identificar el siguiente día hábil fuera de la muestra de entrenamiento
        # Buscamos en el índice de fechas cuáles son posteriores a la fecha de corte actual.
        # 'current_train_end' es el último día usado para entrenar.
        next_days = all_prices.index[all_prices.index > current_train_end]
        #Si no quedan días en el futuro (se acabó el dataset), terminamos el bucle.
        if len(next_days) == 0: break
        # Seleccionamos estrictamente el PRIMER día disponible (t+1).
        # No miramos más allá para no romper la causalidad temporal.
        test_date = next_days[0]
        # Dataset exclusivo para el punto de test
        ds_test = BatchableHybridDataset(
            prices=all_prices.loc[:test_date], tickers=tickers, target_ticker=target_ticker,
            seq_len=seq_len, W=W, 
            train_end_date=current_train_end, horizon_days=1,
            feature_mode="close_minmax", 
            adj_mode="paper",
            price_min=p_min,
            price_max=p_max,# Usamos los mismos escaladores del training (Consistencia) 
            cfg=cfg, split="test"
        )
        test_loader = DataLoader(ds_test, batch_size=1, shuffle=False)
        # Inferencia del Modelo
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                # Mover datos a GPU
                batch = batch.to(device)
                pred_norm = model(batch).item()
                true_norm = batch.y.item() 
                
                preds_norm_list.append(pred_norm)
                true_norm_list.append(true_norm)

                # Desnormalización (Inverse Transform) para obtener precio real
                p_min_t, p_max_t = p_min[target_idx], p_max[target_idx]
                # Calculamos el rango (denominador)
                denom = p_max_t - p_min_t
                if abs(denom) < 1e-8: denom = 1.0
                
                pred_d = pred_norm * denom + p_min_t
                true_d = true_norm * denom + p_min_t
                # Almacenamos resultados en dólares para reportes financieros
                preds_denorm.append(pred_d)
                truth_denorm.append(true_d)
                dates_test.append(test_date)
                
                err = abs(pred_d - true_d)
                print(f"   Predicción: {test_date.date()} | Real: {true_d:.2f} | Pred: {pred_d:.2f} | Diff: {err:.2f}")
        # Avanzar la ventana de entrenamiento al incluir el día recién predicho
        current_train_end = test_date
        print(f"   Paso completado en {time.time()-total_start_time:.2f}s (acum)")

    #CALCULO DE METRICAS FINALES
    preds_n = np.array(preds_norm_list)
    truth_n = np.array(true_norm_list)
    
    if len(preds_n) > 0:
        # MSE Normalizado
        mse_norm = np.mean((preds_n - truth_n)**2)
        # MSE en Dólares
        mse_dollars = np.mean((np.array(preds_denorm) - np.array(truth_denorm))**2)
        
        print(f"\nFINALIZADO {target_ticker}")
        print(f"   MSE (Paper): {mse_norm:.5f}")
        print(f"   MSE ($$$)  : {mse_dollars:.2f}")

        # Generación de gráfico del ticker
        plt.figure(figsize=(10,5))
        plt.plot(dates_test, truth_denorm, label='Real')
        plt.plot(dates_test, preds_denorm, label='Pred')
        plt.title(f"Evaluación Out-of-Sample: {target_ticker} (MSE: {mse_norm:.5f})")
        plt.xlabel("Fecha")
        plt.ylabel("Precio de Cierre")
        plt.legend()
        try:
            plt.savefig(f"Grafico_{target_ticker}.png")
        except:
            pass
        plt.close('all')
        
        return dates_test, np.array(truth_denorm), np.array(preds_denorm), preds_n, truth_n
    else:
        return [], [], [], [], []