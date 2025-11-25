#  **Stocks Forecasting: Hybrid LSTM+GNN**

Este proyecto probó dos enfoques distintos para combinar LSTM + GNN.

## Variante A - LSTM + embeddings GNN como features

### ¿Qué hace?

- Calcula un **embedding GNN** por acción.
- Ese embedding se **agrega como feature** al LSTM (por ej. `[Close, gnn_emb]`).
- Usa `hit_coef` para mejorar la capacidad del modelo de acertar la dirección.

### Creemos que funciono mejor porque…

- El LSTM recibe más información sin complicar la arquitectura.
- La loss direccional ayuda al hit rate.

## Variante B - Fusiona embedding LSTM + embedding GNN en un MLP

### ¿Qué hace?

- Generaba un embedding del LSTM y otro del GNN.
- Los concatenaba.
- Los pasaba por un MLP para predecir.

### Problemas

- No se llegó a optimizar.
- Más ruido, más overfitting.
- Sin `hit_coef`.

La Variante A es la definitiva: simple, estable y con mejores resultados.
