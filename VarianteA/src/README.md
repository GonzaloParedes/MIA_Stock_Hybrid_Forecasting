# LSTM+GNN Hybrid Stock Forecasting

# **1. Objetivo**

Desarrollar un sistema capaz de **predecir el precio futuro de acciones** utilizando una **red neuronal LSTM** entrenada con una estrategia de **walk-forward validation**.

El enfoque busca evaluar el rendimiento del modelo en un escenario realista, reentrenando diariamente con datos históricos y midiendo su capacidad predictiva mediante métricas de error y estrategias de trading (long/short).

# **2. Autoría**

- Luciano Eduardo Rielo, Gonzalo Paredes, David Tobares
- Año: 2025

---

# **3. Etapas**

#### **LSTM**

- [x] **Descarga de datos históricos** desde Yahoo Finance (`tickers_download.py`) para los tickers definidos en `config.yaml`.
- [x] **Preprocesamiento** y almacenamiento automático en CSV de precios ajustados (`Open`, `High`, `Low`, `Close`, `Volume`).
- [x] **Modelo LSTM** (`LSTMRegressor`) con capas recurrentes y lineales. Implementar modelo simple, optimizarlo luego.
- [x] **Implementación de entrenamiento** con early stopping, clipping de gradientes y mini-batches (`train_lstm_model`).
- [x] **Estrategia walk-forward** (`walk_forward_predict`):
    - Reentrena un modelo por cada día de testeo, con el y_real
    - Usa los años previos a el día de testing como set de entrenamiento. 
    - Permite activar o desactivar retroalimentación con predicciones previas.
- [x] **Cálculo de métricas y backtesting**:
    - MSE normalizado.
    - Pérdida o ganancia diaria por dólar invertido (`daily_profit_per_dollar`).
    - Retorno acumulado de los días predichos. 
    - Hit rate de dirección predicha.
- [x] **Visualización** de precios reales vs. predichos, indicando los hits verdaderos y falsos.
- [x]  Incorporar **variables adicionales** (e.g. volumen, indicadores técnicos) para volver el modelo **multivariado**.
- [x] Emprolijar y dejar listo para a futuro optimizar hiperparámetros del LSTM.
- [x] Probar si hay mejoras con indices que tira el chat (pruebo RSI_14 y BB_Width). ***En primera instancia dio peor. ***
- [x] Arreglar que cuando quiero testear un día actual o uno anterior al actual, el método next day me genera un error.
- [x] Expanding window corregido. Antes estaba mal, se desplazaba toda la ventana del trainin set, ahora el training set mantiene el primer día y se alarga el último día a medida que se testea otro, de ahi el "expanding". 
- [x] Se agrego mse_validation, esto se usa para la optimizacion de los hiperparámetros (tuning). Una vez optimizados los hiperparámetros, se procede a entrenar desactivando el mse_validation. El "patience" solo se usa para el tuning.
- [x] Hago tuning de hiperparametros, se reduce el mse significativamente (tuning.ipynb)
- [x] Ver si es mejor optimizar en funcion del hit rate que del closing price
- [x] Agrego grafico que compara el mse con el volumen, agrega correlacion de pearson.
- [x] Optimizar epochs
- [x] Agrego parametro hit_coef el cual aumenta el punishment cuando hace un hit incorrecto.
- [x] Correccion en hit_rate tanto en val_loss como train_loss, se calculaba la diferencia entre días consecutivos del batch. Optimizo el hit_coef. 
- [x] La funcion directional_loss no permite que la direccion de la curva influya en el gradiente. Estaba haciendo todo en vano. Se debe reemplazar por una curva suave. Aplico SoftPlus que si es derivable y compatible con torch para hacer backpropagation. ***Resultados mucho más ruidosos en el test final en AMD, mejoro el hit rate pero disminuyo mucho el mse. Capaz habría que probar validar con un val_set mas chico, o a pesar de que en el tuning el hit_coef optimo de 2, habría que usar uno menor.***
- [x] Probar hit_coef con distintas acciones a ver cual es el optimo para cada una.
- [x] Optimizo con optuna. El MSE parece dar mucho mejor y es mas robusto porque se prueban con varios stocks.
- [x] Hit coef lo reduzco a 0, para maximizar el MSE (por ahora), y utilizo los siguientes hiperparámetros:
        start_day    = "2025-01-01"

#### **GNN**
- [x] Generar relaciones lineales entre nodos. Empiezo haciendo correlaciones de Pearson. Resulta que hay que disminuir el umbral para establecer si hay o no correlación ya que las correlaciones entre stocks disminuyeron con los años. 
- [x] Generar relaciones no lineales entre nodos. 
- [x] Ver como se acopla LSTM con GNN. Los resultados muestran muy buena correlacion entre el MSE y el volúmen aunque no se lo use para entrenar. Hay gran error los días de alto volumen (analisis fundamental)
- [x] Revisar relaciones no lineales y documentar las ecuaciones
- [x] Descargo relaciones no lineales tipo lift, no aportan nada, la aleatoriedad siempre predomina antes la co-dependencia entre stocks 

#### **Detalles finales**

- [x] Generalizacion de resultados para distintos casos con el objetivo de comparar el MSE promedio de todos los stocks
- [x] Hay que comprar, MSE promedio en funcion de si es, LSTM, LSTM+GNN, correlacion en funcion de los años, self-loop, hit_coef, metodo de acoplamiento gnn

### **Pruebas**

- test 1 = "2025-01-01" hiperparametros del paper
- test 2 = hiperparametros optuna (dan mejor) y tau_prior 1.0 que da igual
- test 3 = hit_coef 0.02
- test 4 = hit_coef 0.02 pero solo pearson sin lift (dio mejor)
- test 5 = hit_coef 0 
- test_4_bis = hiperparametros optimizados para "Close" pero sin "hit_coef" optimizado
- 9_hit_coef = hit_coef optimizado para multi feature
- test_6 = multifeature
- Mejores resultados: 6_optuna_test utilizando solo "close" como feature 