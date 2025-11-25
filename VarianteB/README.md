
# Variante A
Este proyecto implementa un modelo de Deep Learning hibrido que combina LSTM (para capturar patrones temporales individuales) y GNN (Graph Neural Networks, para capturar correlaciones entre activos) con el objetivo de predecir precios de acciones.

El entrenamiento utiliza una estrategia de Expanding Window (Ventana Expansiva) para simular un escenario de trading real y evitar fugas de información (data leakage).

## Estructura del Proyecto

* paper.py: Script Principal. Ejecuta este archivo para descargar datos, entrenar el modelo para todos los tickers y generar el reporte final.

* components.py: Define la arquitectura de la red neuronal (HybridModel) y el wrapper de datos para PyTorch Geometric.

* training.py: Contiene la lógica del bucle de entrenamiento paso a paso (día a día) y la validación forward-walk.

* gnn_utils.py: Funciones para construir el grafo dinámicamente basado en la correlación de Pearson y normalizar los features.

* data_utils.py: Configuración global (TrainConfig) y clases base para el procesamiento de ventanas de tiempo.

* plot_heatmap.py: Script auxiliar para generar mapas de calor comparativos de los errores (MSE).
