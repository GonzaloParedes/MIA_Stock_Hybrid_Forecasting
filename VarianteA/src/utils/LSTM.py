import torch.nn as nn

class LSTMRegressor(nn.Module):
    """
    Modelo LSTM simple para predicción de series temporales.  
    La idea es: el LSTM procesa la ventana de datos y del último hidden state saco la predicción final 
    pasándola por una cabeza densa chiquita.

    Inputs (constructor)
    --------------------
    input_size : int
        Cantidad de features que entra en cada paso de tiempo.
    hidden_size : int
        Tamaño del estado oculto del LSTM.
    num_layers : int
        Número de capas LSTM apiladas.
    dropout : float
        Dropout entre capas del LSTM y también en la cabeza final.

    Métodos
    -------
    forward(x)
        Recibe un tensor de shape (batch, window, input_size).  
        Devuelve un tensor (batch, 1) con la predicción del próximo valor.

    Output
    ------
    Predicción escalar por batch, usando solo el último estado del LSTM.
    """
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_layers, 
                 dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Recibe un tensor de shape (batch, window, input_size).  
        Devuelve un tensor (batch, 1) con la predicción del próximo valor.
        """
        out, _ = self.lstm(x)      # (B, W, H)
        last = out[:, -1, :]       # (B, H)
        return self.head(last)     # (B, 1)