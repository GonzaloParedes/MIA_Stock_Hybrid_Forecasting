# replicate_paper.py
import matplotlib
matplotlib.use('Agg') # CRÍTICO: Al inicio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import yfinance as yf
from dateutil.relativedelta import relativedelta
from data_utils import TrainConfig
import gc
import argparse
import traceback

# Importamos la NUEVA función de entrenamiento
from expanding_train_true_batch import expanding_train_true_batch


def inspect_dataset(df):
    """ Función auxiliar para imprimir reporte de calidad de datos """
    print("\n" + "="*60)
    print("AUDITORÍA DE DATOS (PRE-IMPUTACIÓN)")
    print("="*60)
    print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"Rango Fechas: {df.index.min().date()} hasta {df.index.max().date()}")
    
    # 1. Chequeo de Nulos
    null_counts = df.isna().sum()
    total_rows = len(df)
    cols_with_nulls = null_counts[null_counts > 0]
    
    if not cols_with_nulls.empty:
        print(f"\nCOLUMNAS CON DATOS FALTANTES ({len(cols_with_nulls)}):")
        print("-" * 40)
        print(f"{'Ticker':<10} | {'Nulos':<8} | {'% Faltante'}")
        for ticker, count in cols_with_nulls.items():
            pct = (count / total_rows) * 100
            print(f"{ticker:<10} | {count:<8} | {pct:.2f}%")
    else:
        print("\n✅ No hay datos faltantes (Dataset completo).")

    # 2. Chequeo de Vacíos Totales
    empty_cols = null_counts[null_counts == total_rows].index.tolist()
    if empty_cols:
        print(f"\n[CRÍTICO] Tickers 100% vacíos (Revisar descarga): {empty_cols}")
    
    # 3. Vistazo a los datos
    print("\nVISTA PREVIA (HEAD):")
    print(df.head(3))
    print("\nVISTA PREVIA (TAIL):")
    print(df.tail(3))
    print("="*60 + "\n")



def plot_final_grid(store_data, filename="FINAL_REPORT_GRID.png"):
    """
    Genera una imagen con los 10 gráficos de los tickers.
    """
    tickers = sorted(list(store_data.keys()))
    rows = 5
    cols = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 25))
    axes = axes.flatten() 
    
    
    for i, ticker in enumerate(tickers):
        ax = axes[i]
        
        # Si el ticker no está en los resultados o falló
        if ticker not in store_data or store_data[ticker].get('mse') is None:
            ax.text(0.5, 0.5, f"{ticker}: ERROR / NO DATA", ha='center', color='red', fontsize=12)
            continue
            
        data = store_data[ticker]
        dates = data['dates']
        real = data['real']
        pred = data['pred']
        mse = data['mse']       # MSE en Dólares
        mse_n = data['mse_norm'] # MSE Normalizado 
        
        # Graficamos series de tiempo
        ax.plot(dates, real, label='Real ($)', color='black', alpha=0.7, linewidth=1.5)
        ax.plot(dates, pred, label='Pred ($)', color='darkorange', linewidth=1.5)
        
        ax.set_title(f"{ticker} | MSE Paper: {mse_n:.5f} | MSE $: {mse:.2f}", fontsize=11, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Rotamos fechas para mejor lectura en el eje X
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print("Reporte guardado con éxito.")

def main():
    
    PAPER_TICKERS = sorted(['AAPL', 'MSFT', 'CMCSA','COST', 'QCOM','ADBE', 'SBUX', 'INTU', 'AMD', 'INTC'])

    # Fecha de corte para dividir entrenamiento inicial y validación forward-walk
    TRAIN_END = "2025-01-01"

    # Horizonte de predicción: cantidad de días a predecir en el futuro (sliding window)
    HORIZON = 50     
    
    print("Descargando datos")
    t_end = pd.to_datetime(TRAIN_END)
    # Descargamos 2 años hacia atrás para entrenamiento + el horizonte futuro para testeo
    start = t_end - relativedelta(years=2)
    end = t_end + pd.Timedelta(days=HORIZON*2 + 50)

    # Descarga masiva optimizada
    raw = yf.download(PAPER_TICKERS, start=start, end=end, auto_adjust=True, progress=False)
    
    # Normalización de la estructura de datos
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            data_global = raw["Close"]
        else:
            data_global = raw
    except:
        data_global = raw

    # === NUEVO BLOQUE: ANÁLISIS DE DATOS ===
    inspect_dataset(data_global)
    # ==================================
    
    # Imputación simple de valores faltantes
    data_global = data_global.ffill().bfill()

    # === CONFIGURACIÓN ===
    
    cfg = TrainConfig(
        lstm_hidden=64,     # Capacidad de memoria secuencial
        gnn_hidden=32,      # Capacidad de abstracción del grafo
        dropout=0.5,        # Regularización para evitar overfitting
        lr=0.001,           # Learning Rate
        epochs=40,          
        batch_size=11,      # Tamaño de lote pequeño para mayor estocasticidad
        patience=8,         # Early stopping
        pearson_corr=0.5,   # Umbral mínimo de correlación para definir aristas en el grafo
       
    )
    
    store_results = {} # Guardamos cada ticker

    #Iteramos por cada ticker 
    for target in PAPER_TICKERS:
        print(f"\n>>> OBJETIVO ACTUAL: {target} <<<")
        
        # Limpieza de memoria VRAM/RAM antes de cada ciclo pesado
        gc.collect()                  
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        try:
            # Ejecución del entrenamiento
            dates, real, pred, preds_n, true_n = expanding_train_true_batch(
                target_ticker=target,
                tickers=PAPER_TICKERS,
                train_end=TRAIN_END,
                training_years=2,
                horizon_days=HORIZON,
                seq_len=50,     # Longitud de la secuencia de entrada
                W=50,           # Ventana para cálculo de correlaciones
                cfg=cfg,
                preloaded_data=data_global 
            )
            
            if len(preds_n) > 0:
                # Calculo de métricas de error
                mse_norm = np.mean((preds_n - true_n)**2) # MSE sobre datos normalizados
                mse_dollar = np.mean((pred - real)**2)    # MSE en dólares
                
                print(f"MSE {target} (Paper): {mse_norm:.5f}")
                
                # === GUARDADO EN EL DICCIONARIO ===
                store_results[target] = {
                    'dates': dates,
                    'real': real,
                    'pred': pred,
                    'mse': mse_dollar,
                    'mse_norm': mse_norm
                }
                
            else:
                print(f"Sin predicciones para {target}")
                store_results[target] = {'mse': None}
                
        except Exception as e:
            print(f"Error {target}: {e}")
            traceback.print_exc() # Muestra detalle del error
            store_results[target] = {'mse': None}

    # --- Resumen Final ---
    print("\n=== TABLA DE RESULTADOS ===")
    
    valid_mses_norm = []
    print(f"{'Ticker':<10} | {'MSE (Paper)':<12} | {'MSE ($)':<12}")
    print("-" * 40)
    
    for t in PAPER_TICKERS:
        data = store_results.get(t, {})
        mn = data.get('mse_norm')
        md = data.get('mse')
        
        if mn is not None:
            print(f"{t:<10} | {mn:.5f}        | {md:.2f}")
            valid_mses_norm.append(mn)
        else:
            print(f"{t:<10} | FALLÓ| -")
            
    if valid_mses_norm:
        avg = np.mean(valid_mses_norm)
        print("-" * 40)
        print(f"Promedio Global: {avg:.5f}")
        print(f"Meta del Paper: 0.00144")

    # Generación de gráficos
    plot_final_grid(store_results)

if __name__ == "__main__":
    main()