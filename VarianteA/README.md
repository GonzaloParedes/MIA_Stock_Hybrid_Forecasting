# **Stocks Forecasting with LSTM+GNN**

# Objetivo

Este proyecto busca predecir el precio de cierre de acciones usando un modelo híbrido **LSTM + GNN**, a partir de la idea del paper de Sonani et al. La idea es:

- Capturar la dinámica temporal con LSTM.
- Incorporar información del resto del mercado con un grafo de relaciones entre acciones (GNN).
- Evaluar de forma realista con **walk-forward / expanding window**, midiendo MSE, hit rate y PnL.

---

# Estructura de `src/`

```
src/
├── prices/              # CSVs con datos históricos descargados (uno por ticker)
├── tests/               # Notebooks de experimentación y tuning
│   ├── 1_tuning.ipynb
│   ├── 2_window.ipynb
│   ├── 3_epochs.ipynb
│   ├── 4_hit_coef.ipynb
│   ├── 6_optuna_test.ipynb
│   ├── 7_hit_coef.ipynb
│   ├── 8_optuna_test.ipynb
│   ├── 9_hit_coef.ipynb
│   ├── test.ipynb, test2.ipynb, ...
│   └── test6.ipynb      # Varios tests puntuales de ideas/escenarios
├── utils/
│   ├── gnn.py           # Construcción del grafo y embeddings GNN (Pearson + lift)
│   ├── LSTM.py          # Clase LSTMRegressor y funciones de entrenamiento/walk-forward
│   ├── metrics.py       # Cálculo de MSE, hit rate, PnL diario, tablas comparativas, etc.
│   ├── plots.py         # Funciones para graficar resultados de forecasting y curvas de capital
│   └── utils.py         # Helpers generales: descarga de tickers, trading days, utils varios
├── config.yaml          # Configuración: tickers, rango de fechas, parámetros por defecto, etc.
└── main.ipynb           # Notebook principal para correr predicciones

```

# **Breve descripción de los archivos**

- **`utils/gnn.py`**
    
    Arma la matriz de pesos entre acciones usando correlación de Pearson y co-movimiento (lift), normaliza la matriz y genera los **embeddings GNN** que después se le pasan al LSTM cuando se usa el modo `LSTM+GNN`.
    
- **`utils/LSTM.py`**
    
    Define el modelo `LSTMRegressor` 
    
- **`utils/metrics.py`**
    
    Centraliza métricas: MSE, hit rate, PnL por dólar, retorno acumulado, etc.
    
    También tiene funciones para resumir resultados de un experimento.
    
- **`utils/plots.py`**
    
    Funciones de ploteo: curvas real vs predicho, marcadores de dirección OK/FAIL, heatmaps por modelo y por stock, etc.
    
- **`utils/utils.py`**
    
    Utilidades generales. Por ejemplo:
    
    - `dowload_tickers(...)` para descargar datos desde yfinance y guardarlos en `./prices`.
    - construcción de ventanas (`build_xy_from_series`),
    - entrenamiento con early stopping (`train_lstm_model`),
    - predicción walk-forward (`walk_forward_predict`),
    - helpers como `next_trading_day`, `previous_trading_day`, etc.
- **`tests/*.ipynb`**
    
    Notebooks de prueba: tuning de hiperparámetros (window, epochs, hit_coef, Optuna, etc.) y distintos experimentos intermedios.
    

# `main.ipynb`: flujo principal para predecir

`main.ipynb` es el notebook “de uso diario”. La idea es que desde ahí puedas elegir:

- el **ticker** que querés predecir,
- la **fecha de predicción**,
- si usás **LSTM** o **LSTM+GNN**,
- y los hiperparámetros básicos (ventana, hidden_size, hit_coef, etc.).

El flujo típico dentro del notebook es algo así:

1. **Carga de configuración**
    
    Lee `config.yaml` (tickers, rango de datos) y/o define los parámetros en celdas.
    
2. **Carga / descarga de datos**
    - Si ya existen CSVs en `./prices`, los lee.
    - Si no, usa `dowload_tickers()` para descargarlos y guardarlos.
3. **Selección de modelo y parámetros**
    
    Elegís:
    
    - `model = "LSTM"` o `"LSTM+GNN"`,
    - valores de `W`, `hidden_size`, `num_layers`, `hit_coef`, etc.
4. **Predicción**
    - Para un horizonte de varios días: llama a `walk_forward_predict(...)`
    - Para un solo día puntual (por ejemplo “mañana”) simplemente fijar el horizonte en 1 día.
5. **Análisis de resultados**
    - Usa funciones de `metrics.py` y `plots.py` para:
        - ver MSE, hit rate, PnL medio,
        - graficar curva real vs predicha,
        - ver en qué días el modelo acierta o falla la dirección.

La idea es que no tener que tocar el código de las utils: solo cambiar parámetros en `main.ipynb`, correr las celdas y analizar cómo se comporta el modelo para el stock y la fecha que te interese.