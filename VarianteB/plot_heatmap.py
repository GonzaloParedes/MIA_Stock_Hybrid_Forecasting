import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def generar_heatmap_comparativo():
    # ---------------------------------------------------------
    # 1. PREPARACIÓN DE DATOS
    # ---------------------------------------------------------
    
    # Lista de Tickers ordenados alfabéticamente para consistencia
    tickers = sorted(['AAPL', 'ADBE', 'AMD', 'CMCSA', 'COST', 'INTC', 'INTU', 'MSFT', 'QCOM', 'SBUX'])
    
    # DATOS 1: LSTM (Línea Base)
    # NOTA: He copiado estos valores visualmente de la IMAGEN que subiste (fila superior).
    # Si tienes tus propios resultados del benchmark LSTM, reemplaza estos valores.
    lstm_data = {
        'AAPL': 0.0017, 'MSFT': 0.0015, 'CMCSA': 0.0048, 'COST': 0.0030, 
        'QCOM': 0.0011, 'ADBE': 0.0013, 'SBUX': 0.0031, 'INTU': 0.0015, 
        'AMD': 0.0006, 'INTC': 0.0013
    }

    # DATOS 2: TU MODELO (LSTM + GNN)
    # Estos son los datos copiados de tu tabla de texto provista.
    my_model_data = {
        'AAPL':  0.01099,
        'ADBE':  0.00286,
        'AMD':   0.00155,
        'CMCSA': 0.00962,
        'COST':  0.04106, # Este valor es muy alto, dominará la escala de color
        'INTC':  0.00518,
        'INTU':  0.00182,
        'MSFT':  0.00373,
        'QCOM':  0.01063,
        'SBUX':  0.01025
    }

    # Crear listas ordenadas
    data_lstm = [lstm_data[t] for t in tickers]
    data_gnn  = [my_model_data[t] for t in tickers]

    # Calcular promedios para el título
    avg_lstm = np.mean(data_lstm)
    avg_gnn = np.mean(data_gnn)

    # Crear DataFrame para Seaborn
    df = pd.DataFrame([data_lstm, data_gnn], 
                      columns=tickers, 
                      index=["LSTM (Baseline)", "Tu Modelo (LSTM+GNN)"])

    # ---------------------------------------------------------
    # 2. GENERACIÓN DEL GRÁFICO
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 4)) # Formato ancho como el de la imagen
    
    # Usamos coolwarm (Azul=Bajo Error, Rojo=Alto Error)
    # vmin y vmax ayudan a que el contraste sea visible, dado que COST tiene un error muy alto (0.04)
    # ajustamos vmax un poco para que no todo lo demás se vea azul oscuro.
    ax = sns.heatmap(df, 
                     annot=True,       # Escribir los números
                     fmt=".5f",        # 5 decimales
                     cmap="coolwarm",  # Mapa de color Azul-Rojo
                     linewidths=.5,    # Líneas entre celdas
                     cbar_kws={'label': 'MSE (Normalizado)'})

    # Título dinámico
    plt.title(f"MSE por modelo y stock | LSTM avg={avg_lstm:.5f} • LSTM+GNN avg={avg_gnn:.5f}", fontsize=14)
    
    # Ajustes de etiquetas
    plt.xlabel("Activos (Stocks)", fontsize=11)
    plt.ylabel("Modelo", fontsize=11)
    
    # Rotar los nombres de los tickers para que se lean bien
    plt.xticks(rotation=45)
    plt.yticks(rotation=0) 
    
    plt.tight_layout()
    
    # Guardar y Mostrar
    filename = "comparacion_heatmap_final.png"
    plt.savefig(filename, dpi=300)
    print(f"✅ Gráfico guardado como: {filename}")
    plt.show()

if __name__ == "__main__":
    generar_heatmap_comparativo()