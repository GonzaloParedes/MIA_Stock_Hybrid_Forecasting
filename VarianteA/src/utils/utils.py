import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm  # Import tqdm for progress bar
import yfinance as yf
import pandas as pd
from copy import deepcopy
from utils.gnn import *
from utils.metrics import * 

def dowload_tickers(config_data):
    """
    Descarga precios históricos de varios tickers usando yfinance según la configuración
    pasada, arma un dataframe por cada acción y los guarda en ./prices/.  
    Devuelve todos los dataframes listos para usar.

    Inputs
    ------
    config_data : dict
        Diccionario con la estructura:
        {
            "ticker_downloads": {
                "tickers": [...],
                "start": "YYYY-MM-DD",
                "end":   "YYYY-MM-DD"
            }
        }

    Lo que hace
    -----------
    - Descarga los precios desde Yahoo Finance.
    - Separa cada ticker en su propio DataFrame con columnas:
    ["Open", "High", "Low", "Close", "Volume"].
    - Muestra advertencias si la fecha disponible es distinta a la pedida.
    - Guarda los CSV en ./prices/.

    Output
    ------
    dfs : dict
        Diccionario {ticker: dataframe} con los precios descargados y limpios.
    """

    # Obtener los valores de tickers, start y end
    tickers = config_data['ticker_downloads']['tickers']
    start = config_data['ticker_downloads']['start']
    end = config_data['ticker_downloads']['end']
    # Descarga del dataset
    prices = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)

    print(f"Tickers descargados {tickers}.\n")

    # separo las columnas de cada ticker en distintos dataframes.
    dfs = {ticker: prices.xs(ticker, axis=1, level=1)[["Open", "High", "Low", "Close", "Volume"]].copy() for ticker in tickers}

    if pd.Timestamp(start) != dfs[tickers[0]].index.min():
        print(f"Advertencia:\nFecha de inicio solicitada: ({start})\nFecha mínima disponible: ({dfs[tickers[0]].index.min().date()})\nSe usará la fecha mínima disponible.\n")
    if pd.Timestamp(end) != dfs[tickers[0]].index.max():
        print(f"Advertencia:\nFecha de fin solicitada ({end})\nFecha máxima disponible ({dfs[tickers[0]].index.max().date()})\nSe usará la fecha máxima disponible.\n")

    # guardo los archivos en un csv
    for ticker in tickers:
        dfs[ticker].to_csv(f"./prices/{ticker}_prices.csv")

    print("Descarga completada.")
    return dfs

def build_xy_from_series(series: np.ndarray, window: int):
    """
    Convierte una serie temporal en pares (X, y) usando una ventana deslizante.  
    Básicamente arma los ejemplos que después usa el LSTM: cada X es una ventana 
    de `window` pasos y el y es el valor real en el paso siguiente.

    Inputs
    ------
    series : np.ndarray
        Serie temporal con shape (T,) o (T, F).  
        Si viene 1D, se convierte automáticamente a (T, 1).
    window : int
        Tamaño de la ventana que se usa para predecir el siguiente valor.

    Outputs
    -------
    X : np.ndarray
        Arreglo con shape (N, window, F), donde N = T - window.
    y : np.ndarray
        Targets con shape (N,), usando la primera feature como objetivo.

    Notas
    -----
    Si la serie es demasiado corta para armar una sola ventana, devuelve arrays vacíos.
    """


    if series.ndim == 1:
        series = series.reshape(-1, 1)  # (T,1)

    X, y = [], []
    for t in range(window, len(series)):
        X.append(series[t-window:t])      # (window, F)
        y.append(series[t, 0])            # target = Close en t (feature 0)

    if not X:
        F = series.shape[1]
        return (np.empty((0, window, F), np.float32),
                np.empty((0,), np.float32))
    return np.array(X, np.float32), np.array(y, np.float32)

def next_trading_day(day, idxs):
    """
    Devuelve el próximo día hábil a partir de una fecha dada.  
    Va sumando días hasta encontrar uno que figure en el índice (idxs).

    Inputs
    ------
    day : pd.Timestamp
        Fecha desde la cual quiero avanzar.
    idxs : pd.DatetimeIndex
        Fechas hábiles disponibles (generalmente el índice del DataFrame de precios).

    Output
    ------
    pd.Timestamp
        El siguiente día hábil después de `day`.
    """

    finded = False
    while not finded:
        day = day + pd.Timedelta(days=1)
        if day in idxs:
            finded = True
    return day 

def previous_trading_day(day, idxs):
    """
    Devuelve el día hábil anterior a una fecha dada.  
    Básicamente va restando días hasta encontrar uno que exista en el índice (idxs).

    Inputs
    ------
    day : pd.Timestamp
        Fecha desde la cual quiero retroceder.
    idxs : pd.DatetimeIndex
        Conjunto de fechas hábiles disponibles (por lo general el índice del DataFrame de precios).

    Output
    ------
    pd.Timestamp
        El día hábil inmediatamente anterior a `day`.
    """

    finded = False
    while not finded:
        day = day - pd.Timedelta(days=1)
        if day in idxs:
            finded = True
    return day 
import torch.nn.functional as F

@torch.no_grad()
def evaluate_val_loss(model, val_loader, device, hit_coef: float):
    """
    Evalúa el loss de validación del modelo usando el loader de validación.  
    Calcula el loss total (ponderado por tamaño de batch) usando la misma 
    directional_loss que se usa en el entrenamiento.

    Inputs
    ------
    model : nn.Module
        Modelo ya en modo eval().
    val_loader : DataLoader
        Loader con los datos de validación (X_val, y_val).
    device : torch.device
        cpu o gpu, según dónde esté corriendo el modelo.
    hit_coef : float
        Peso de la penalización por dirección en la función de pérdida.

    Output
    ------
    float
        El loss promedio de validación.  
        Si no hay datos, devuelve inf.
    """

    if val_loader is None:
        return float("inf")
    model.eval()

    total_loss = 0.0
    total_n = 0

    for xb, yb in val_loader:
        xb = xb.to(device); yb = yb.to(device)      # xb: (B,W,F), yb: (B,1)
        pred = model(xb)                            # (B,1)

        # último Close de la ventana, feature 0
        last_close = xb[:, -1, 0].unsqueeze(1)      # (B,1)

        loss = directional_loss(pred, yb, last_close, hit_coef)  # loss "mean" del batch
        bsz = yb.size(0)
        total_loss += loss.item() * bsz             # acumular ponderado por batch size
        total_n    += bsz

    return total_loss / total_n if total_n > 0 else float("inf")


def directional_loss(pred, target, last_close, hit_coef=0.5, k = 10.0):
    """
    Combina MSE con penalización si la predicción va en la dirección incorrecta.
    """
    
    # penalización de dirección incorrecta
    """ cuando se hace target[1:] - target[:-1] se restan los valores de i+1 e i pero de cada elemento del batch.
    Esto significa que se está calculando un promedio de hits comprando la muestra i con la muestra i+1 del batch,
    generando pares de muestras consecutivas dentro del batch, NO de muestras temporalmente consecutivas.
    Aunque el batch se genere con shuffle=False, y el dataset estuviera ordenado cronologicamente, target[i] y target[i+1]
    son objetivos de dos muestras distintas (tienen los dos ventanas distintas).
    En resumen, se calcula un hit de días que no son consecutivos. 
    """
    # direction_real = torch.sign(target[1:] - target[:-1])
    # direction_pred = torch.sign(pred[1:] - pred[:-1])

    
    # MSE normal
    mse = F.mse_loss(pred, target)
    # Penalizacion suave: valores grandes margin < 0, aprox 0 si margin > 0
    margin = (pred - last_close) * (target - last_close)
    dir_term = F.softplus(-k * margin).mean()

    return mse + hit_coef * dir_term


def train_lstm_model(lstm, 
                    Xtr, ytr,

                    input_size=2, 
                    hidden_size=64, 
                    num_layers=2, 
                    dropout=0.5,

                    hit_coef = 0,
                    batch_size=11, 
                    epochs=40, 
                    lr=0.005,
                    patience=5,
                    torch_seed = None):
    """
    Entrena un modelo LSTMRegressor usando los datos de entrenamiento actuales.  
    Incluye validación, early stopping y la opción de usar una función de pérdida con 
    penalización por dirección (hit_coef). Devuelve el modelo con los mejores pesos 
    según el val_loss.

    Inputs
    ------
    lstm : class
        Clase del modelo LSTM (por lo general LSTMRegressor).
    Xtr : np.ndarray
        Features de entrenamiento con shape (N, W, F).
    ytr : np.ndarray
        Targets reales normalizados, shape (N,).

    input_size : int
        Cantidad de features por timestep.
    hidden_size : int
        Tamaño del hidden state del LSTM.
    num_layers : int
        Número de capas LSTM apiladas.
    dropout : float
        Dropout tanto dentro del LSTM (si num_layers > 1) como en la cabeza del modelo.

    hit_coef : float
        Peso que le damos a la penalización por dirección equivocada en la loss.
        Si es 0, la loss se comporta similar a un MSE normal.
    batch_size : int
        Tamaño del minibatch.
    epochs : int
        Cantidad máxima de épocas a entrenar.
    lr : float
        Learning rate del optimizador Adam.
    patience : int
        Número de épocas sin mejorar el val_loss antes de frenar con early stopping.
    torch_seed : int o None
        Semilla para reproducibilidad. Si es None, no fija seed.

    Que hace
    -----------
    - Divide el dataset en train y validation.
    - Entrena el LSTM usando Adam y gradient clipping.
    - Evalúa val_loss cada época.
    - Guarda los mejores pesos según la validación.
    - Aplica early stopping cuando deja de mejorar.
    - Al final devuelve el modelo con los mejores pesos guardados.

    Output
    ------
    model : nn.Module
        El modelo entrenado, con los mejores pesos encontrados durante el entrenamiento.
    """


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # datasets -> train y validation
    N = len(Xtr)

    val_count = int(N*0.2)

    train_count = N - val_count
    val_count = N - train_count # Reajustar val para no dejarlo en 0

    # train set
    X_tr, y_tr = Xtr[:train_count], ytr[:train_count]
    X_val, y_val = Xtr[train_count:], ytr[train_count:]
    ds_val = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).view(-1,1))
    val_loader = DataLoader(ds_val, batch_size=len(ds_val), shuffle=False, drop_last=False)

    # data -> tensor
    ds_tr  = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).view(-1,1))
    train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False) # shuffle True quita el orden temporal en el batch

    # modelo LSTM, criterio y optimizador
    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    model = lstm(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
                                ).to(device)
    
    criterion = lambda pred, target, last_close: directional_loss(pred, target, last_close, hit_coef)
    # criterion = nn.MSELoss()  # computa por defecto el 'mean' MSE del batch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # early stopping, queremos el modelo con:
        # best_loss = el MSE más bajo
        # best_state = los pesos de ese modelo
        # wait = cuantas epocas llevamos sin mejorar
    best_val, best_state, bad_epochs  = float("inf"), None, 0
    
    # bucle de entrenamiento por epoca
    for ep in range(1, epochs+1): # 
        model.train()  # pone al modelo en modo entrenamiento

        for xb, yb in train_loader:  # se itera cada minibatch dl = [batch1, batch2, ...] donde batch = (xb, yb)
            xb, yb = xb.to(device), yb.to(device)   # los mueve a la GPU si es posible
            optimizer.zero_grad()   # limpia los gradientes del optimizador
            pred = model(xb)        # prediccion
            last_close = xb[:, -1, 0].unsqueeze(1)
            loss = criterion(pred, yb, last_close) # calcula el loss (MSE)
            loss.backward()            # backpropagation para calcular los gradientes y los deja en param.grad
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clipping de gradientes para evitar exploding gradients por norma L2 a 1.0
            optimizer.step()           # actualiza los parámetros de Adam usando los gradientes actuales 

        val_loss = evaluate_val_loss(model, val_loader, device, hit_coef=hit_coef)
        # Early stopping en base a val_loss
        if val_loss + 1e-9 < best_val:
            best_val   = val_loss
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience is not None:
                if bad_epochs >= patience:
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    break

    
    # al final de las epochs, cargo los mejores pesos en el modelo y lo retorno
    if best_state is not None:
        model.load_state_dict(best_state)
        
    return model


def walk_forward_predict(lstm,
                        dfs, 
                        stock,
                        start_day, 
                        horizon_days,
                        training_years,
                        feat_cols:list,

                        W:int,
                        
                        hidden_size, 
                        num_layers, 
                        dropout,
                        hit_coef:float,
                        batch_size:int, 
                        epochs: int, 
                        lr, 
                        patience:int,
                        torch_seed,

                        model:str = 'LSTM+GNN',

                        tau_pear = 0.7,
                        tau_lift = 1.7,
                        alpha = 0,
                        retroalimentacion:bool=False):
    """
    Hace predicciones tipo walk-forward: para cada día de test entrena un LSTM nuevo usando 
    los últimos `training_years` años, arma la ventana W, normaliza, predice y guarda todo.  
    Opcionalmente también mete embeddings GNN (para el modo LSTM+GNN) y permite 
    retroalimentación diaria usando las predicciones previas.

    Es básicamente la función principal del pipeline: entrena-predice-avanza, día por día.

    Inputs
    ------
    lstm : class
        Clase del modelo LSTM (por ejemplo LSTMRegressor).
    dfs : dict
        Diccionario {ticker: dataframe de precios}.
    stock : str
        Acción específica a predecir.
    start_day : str o Timestamp
        Primer día de test.
    horizon_days : int
        Cantidad de días hábiles a predecir en modo walk-forward.
    training_years : int
        Cantidad de años hacia atrás que se usan para entrenar antes de cada día de test.
    feat_cols : list
        Features base a usar (por ej. ["Close"]).

    W : int
        Ventana temporal usada por el LSTM.
    hidden_size, num_layers, dropout : hiperparámetros del LSTM.
    hit_coef : float
        Peso del término direccional en la loss.
    batch_size, epochs, lr, patience : hiperparámetros del entrenamiento LSTM.
    torch_seed : int
        Semilla para reproducibilidad.

    model : str
        "LSTM" o "LSTM+GNN".  
        Si es "LSTM+GNN", se generan embeddings GNN externos y se agregan como feature.
    tau_pear, tau_lift, alpha : floats
        Parámetros para construir el grafo (Pearson + co-movimiento).
    retroalimentacion : bool
        Si es True, la ventana W usa predicciones pasadas en vez de datos reales cuando corresponde.

    Lo que hace
    -----------
    - Calcula los días de test (solo días hábiles).
    - Si corresponde, calcula embeddings GNN para todo el rango global de entrenamiento.
    - Para cada día de test:
        * arma el rango de entrenamiento (training_years),
        * normaliza en base a ese rango,
        * construye Xtr, ytr,
        * arma la ventana Xte del día que se va a predecir,
        * aplica retroalimentación si corresponde,
        * entrena un modelo LSTM desde cero,
        * predice el precio,
        * guarda los resultados,
        * actualiza métricas (MSE, hit rate).

    Output
    ------
    res : tuple
        (res_df, data)

        res_df : DataFrame indexado por día de test, con:
            - y_true
            - y_pred
            - y_true_norm
            - y_pred_norm
            - cscale, cmin   (valores del min-max usado)

        data : dict con todos los parámetros del experimento y métricas finales:
            - mse
            - hit_rate
            - test_days
            - hiperparámetros
            - si usó GNN o no
            - etc.

    Notas
    -----
    Es el corazón del experimento walk-forward: cada día se entrena un modelo nuevo 
    para evitar leakage y mantener el escenario más realista posible.
    """

    
    if model == 'LSTM+GNN':
        use_gnn = True  
    elif model=='LSTM':
        use_gnn = False

    # device 'cpu' o 'gpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = dfs[stock].sort_index()
    idxs = df.index

    # si el horizonte es 2, y los siguientes 2 días al primer test day no son hábiles, los saltea y toma los siguientes 2 habiles.
    current_day = pd.Timestamp(start_day)

    # compruebo que current_day exista en el training_set
    if current_day not in idxs:
        current_day = next_trading_day(current_day, idxs)

    test_days = [current_day]
    for _ in range(horizon_days-1):
        current_day = next_trading_day(current_day, idxs)
        test_days.append(current_day)

    assert len(test_days)==horizon_days, "No hay suficientes días hábiles para testear en el horizonte dado."

     
    test_days = pd.DatetimeIndex(test_days) 

    pred_overrides = {}  # fecha -> y_pred (escala real) para usar en la ventana del siguiente día (retroalimentación con predicciones) 
    out = []             # (day, y_true, y_pred)
    
    ## TRAINING
    ###########

    global_start = test_days[0] - pd.DateOffset(years=training_years)
    global_end   = test_days[-1] - pd.DateOffset(days=1)

    # 1 sola llamada: grafo + embeddings para TODO el rango
    if use_gnn:
        Z_global, gnn_dates, gnn_stocks, _, _ = create_gnn_embeddings(
            dfs, global_start, global_end, ["Close"],                              
            tau_pear = tau_pear,
            tau_lift = tau_lift,
            alpha = alpha,
        )
        stock_idx = gnn_stocks.index(stock)
        gnn_series_global = pd.Series(Z_global[:, stock_idx, 0], index=gnn_dates)
    else:
        gnn_series_global = None

    # bucle por cada día de test
    train_start = test_days[0] - pd.DateOffset(years=training_years)
    for day in test_days:
        # train en 'training_years' años previos (sin overrides para evitar leakage en min-max)
        # fechas de train en formato pd.Timestamp 
        train_end   = day - pd.DateOffset(days=1)
        # train dataframe
        train_df = df.loc[train_start:train_end].copy()
        
        # si no hay suficientes datos para entrenar, saltea el test day
            # esto para verificar que exista al menos un ejemplo de entrenamiento
        if len(train_df) < W + 1:
            continue

        # min-max del train
        eps = 1e-12 # para evitar la division por 0

        if use_gnn:
            gnn_series_train = gnn_series_global.loc[train_start:train_end]
            # alineamos con índice de train_df y rellenamos posibles huecos
            gnn_series_train = gnn_series_train.reindex(train_df.index).ffill().bfill()

            # añadimos columna al train_df
            train_df["gnn_emb"] = gnn_series_train.values

            # para usar la columna de gnn (ya esta normalizada)
            feat_cols_gnn = feat_cols + ["gnn_emb"]   # por ej. ["Close", "gnn_emb"]
        else:
            feat_cols_gnn = feat_cols

        # if not scaled:
        cmins   = train_df[feat_cols_gnn].min().astype(float).values          # shape (2,)
        cmaxs   = train_df[feat_cols_gnn].max().astype(float).values          # shape (2,)
        cscales = np.maximum(cmaxs - cmins, eps)                          # evita división por cero

        train_norm = ((train_df[feat_cols_gnn].values - cmins) / cscales).astype(np.float32)  # (T,F)

        Xtr, ytr = build_xy_from_series(train_norm, W)  

        # ventana W para para el día de test
            # selecciona los W de días antes del día de test
        window_dates = df.loc[: day - pd.tseries.offsets.BDay(1)].tail(W).index
            # valores REALES de los window_dates
        past_close = df.loc[window_dates, feat_cols].astype(float).values.copy()   # (W,F)
            # Embeddings GNN para esas fechas
        
        if use_gnn:
            gnn_series_window = gnn_series_global.reindex(window_dates).ffill().bfill().values.reshape(-1, 1) # (W,F+1)

                # Matriz completa de ventana: [Close, gnn_emb]
            past_window = np.concatenate([past_close, gnn_series_window], axis=1)  # (W,F+1)
        else:
            past_window = past_close.copy()
        
        # Bloque de RETROALIMENTACION: reemplaza en la ventana los valores que ya fueron predichos
        if retroalimentacion:
            for i, d in enumerate(window_dates):
                if d in pred_overrides:
                    past_window[i, 0] = pred_overrides[d]   # sólo Close
        
        # normalizo la ventana de test con el min-max del train
        Xte = (((past_window - cmins) / cscales).astype(np.float32)).reshape(1, W, len(feat_cols_gnn))

        # entrenar y predecir
        model = train_lstm_model(
                                lstm, 
                                Xtr, ytr,
                                input_size=len(feat_cols_gnn), 
                                hidden_size=hidden_size, 
                                num_layers=num_layers, 
                                dropout=dropout,
                                hit_coef=hit_coef,
                                batch_size=batch_size, 
                                epochs=epochs, 
                                lr=lr,
                                patience=patience,
                                torch_seed = torch_seed)
        
        model.eval() # modo evaluación
        with torch.no_grad(): # el with es para no calcular gradientes en esta sección
            # predicción normalizada
            yhat_norm = model(torch.tensor(Xte, dtype=torch.float32, device=device)).squeeze().item()
        y_pred = yhat_norm * cscales[0] + cmins[0]     # feature 0 = Close
        y_true = float(df.loc[day, "Close"])
        y_true_norm = (y_true - cmins[0]) / cscales[0]
        
        # almaceno resultados
        out.append((day, y_true, y_pred, y_true_norm, yhat_norm, cscales[0], cmins[0]))
        pred_overrides[day] = y_pred                  # retroalimentación para el próximo día

        data = {
                "df": df, 
                "start_day": start_day, 
                "horizon_days": horizon_days,
                "training_years": training_years,
                "feat_cols": feat_cols_gnn,
                "test_days": test_days,

                "W": W,
                
                "hidden_size": hidden_size, 
                "num_layers": num_layers, 
                "dropout": dropout,
                "hit_coef": hit_coef,
                "batch_size": batch_size, 
                "epochs": epochs, 
                "lr": lr, 
                "patience": patience,
                "torch_seed": torch_seed,

                "use_gnn": use_gnn,
                "retroalimentacion": retroalimentacion}
        
        res = (pd.DataFrame(out, columns=["day", "y_true", "y_pred", "y_true_norm", "y_pred_norm", "ccsale", "cmin"]).set_index("day"), data)
        mse, hit_rate = metrics(res)
        res[1]["mse"] = mse
        res[1]["hit_rate"] = hit_rate

    return res

def add_today(  df,
                close_price,
                volumen= np.nan,
                open= np.nan, 
                high= np.nan, 
                low= np.nan):
    """
    Agrega un nuevo día al final del DataFrame de precios.  
    Sirve para cuando quiero sumar el precio de hoy de forma manual para predecir mañana.

    Inputs
    ------
    df : pd.DataFrame
        DataFrame original con las columnas clásicas: ["Open", "High", "Low", "Close", "Volume"].
    close_price : float
        Precio de cierre del nuevo día.
    volumen : float (opcional)
        Volumen negociado del día. Si no lo paso, queda como NaN.
    open, high, low : float (opcionales)
        Precios del día (apertura, máximo, mínimo). Si no los paso, quedan como NaN.
    """
    
    last_date = df.index.max() 
    new_date = last_date + pd.Timedelta(days=1)
    df.loc[new_date] = [open, high, low, close_price, volumen]