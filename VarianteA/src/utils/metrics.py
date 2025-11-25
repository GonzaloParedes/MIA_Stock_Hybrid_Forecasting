import numpy as np
import pandas as pd
import pandas as pd
from copy import deepcopy
from utils.gnn import *

def metrics(
        res):
    """
    Calcula las métricas principales del modelo: el MSE entre las series normalizadas
    y el hit rate real (qué porcentaje de días el modelo acertó la dirección del precio).
    Es básicamente un resumen rápido de cómo rindió el modelo en el periodo de test.

    Inputs
    ------
    res : tuple
        Tupla que devuelve la función de forecasting:
        - res[0]: DataFrame con y_true, y_pred, y_true_norm, y_pred_norm.
        - res[1]: diccionario con datos del experimento, incluyendo el df original.

    Outputs
    -------
    mse : float
        Error cuadrático medio entre las predicciones normalizadas y los valores reales.
    hit_rate : float
        Porcentaje de aciertos en la dirección del movimiento del precio.
    """
    res_df = res[0]
    data = res[1]
    df = data["df"]

    y_true = res_df["y_true"].copy()
    y_pred = res_df["y_pred"].copy()

    # metricas
    mse = float(np.mean((res_df["y_pred_norm"] - res_df["y_true_norm"])**2))
    out, mean_daily, hit_rate, hits = daily_profit_per_dollar(res_df, df)
    pnls = out["pnl_per_$"].values
    retorno_acumulado = np.exp(np.sum(np.log(1 + pnls))) - 1

    # Día hábil anterior al inicio del test
    test_start = res_df.index[0]
    prev_day = df.index[df.index < test_start].max()
    y_prev = y_true.shift(1)
    y_prev.iloc[0] = df.loc[prev_day, "Close"]

    # Cálculo de dirección
    r_real = (y_true - y_prev) / y_prev
    r_pred = (y_pred - y_prev) / y_prev
    hit = np.sign(r_pred) == np.sign(r_real)
    hit_rate = hit.mean() * 100

    return mse, hit_rate

def daily_profit_per_dollar(
    res_df: pd.DataFrame,
    df: pd.DataFrame,
    price_col: str = "Close",
    allow_short: bool = True,
    tau: float = 0,        # umbral opcional para evitar operar ruido
) -> pd.DataFrame:
    
    """
    Calcula el PnL diario por cada dólar apostado a lo largo del periodo de test y arma la
    curva de capital simulada, asumiendo que cada día opero según la dirección que predice el modelo.
    Además devuelve el hit rate y el vector de aciertos día a día.

    Inputs
    ------
    res_df : pd.DataFrame
        DataFrame de resultados del modelo. Debe tener las columnas ["y_true", "y_pred"]
        y usar como índice las fechas de test (DatetimeIndex).
    df : pd.DataFrame
        DataFrame original con la serie completa de precios (incluye los días previos al test).
    price_col : str
        Nombre de la columna de precios a usar como referencia (por defecto "Close").
    allow_short : bool
        Si es True, se permiten posiciones en corto (s = -1). Si es False, solo se toman
        posiciones largas (s = 1) cuando la predicción es alcista.
    tau : float
        Umbral mínimo sobre el retorno predicho para decidir operar. Si el módulo del retorno
        predicho es menor a tau, ese día no se opera (s = 0).

    Outputs
    -------
    out : pd.DataFrame
        DataFrame indexado por días de test (o subset de ellos) con:
            - "signal": señal tomada ese día (+1, -1, 0)
            - "pnl_per_$": PnL diario por cada dólar apostado.
    mean_daily : float
        Ganancia media diaria por dólar (promedio de pnl_per_$).
    hit_rate : float
        Proporción de días con PnL positivo (entre 0 y 1).
    hits : list[int]
        Lista de 1/0 indicando si ese día se ganó (1) o no (0).
    """


    # test_days en formato pd.DatetimeIndex
    test_days = res_df.index
    # columna de valor de cierre en df
    prices = df[price_col]

    pnls = []
    signals = []
    hits = []

    # para cada día de test, usamos el precio del día anterior en df
    for day in test_days:
        # ubico el índice del día en df y tomamos el anterior (asumimos df tiene días hábiles)
        loc = prices.index.get_loc(day)
        # indice del día anterior
        prev_idx = loc - 1

        if prev_idx < 0: # por las dudas
            # no hay día anterior en df → no podemos evaluar este día
            print(f"NO HAY DIA ANTERIOR PARA {day.date()}, SE SALTEA")
            continue
        
        # precios real previo real (t-1), predicho (t), real (t)
        p_prev = float(prices.iloc[prev_idx])             # P_{d-1}
        p_pred = float(res_df.at[day, "y_pred"])          # \hat P_d
        p_true = float(res_df.at[day, "y_true"])          # P_d (real)

        # retornos real y predicho (sobre P_{d-1})
        r_real = (p_true - p_prev) / p_prev
        r_pred = (p_pred - p_prev) / p_prev

        # suponemos decisiones basadas en la prediccion
            # señal s = +1 -> compro 1 USD porque el modelo predijo que subiría
            # señal s = -1 -> vendo 1 USD porque el modelo predijo que bajaría
            # señal s = 0 -> no hago nada
                # s nos dice en que direccion apostamos
        if allow_short:
            s = 1.0 if r_pred >  tau else (-1.0 if r_pred < -tau else 0.0)
        else:
            s = 1.0 if r_pred >  tau else 0.0

        # pnl = perdida o ganancia por dolar apostado
            # si s = +1 y r_real > 0 -> ganancia
            # si s = +1 y r_real < 0 -> perdida
            # si s = -1 y r_real > 0 -> perdida
            # si s = -1 y r_real < 0 -> ganancia  
        pnl = s * r_real  # ganancia/pérdida por $1 ese día
        if pnl > 0:
            hits.append(1)
        else:
            hits.append(0)
            
        pnls.append(pnl)
        signals.append(s)

    hit_rate = np.sum(hits)/len(hits)
    out = pd.DataFrame({"signal": signals, "pnl_per_$": pnls}, index=test_days[:len(pnls)])
    mean_daily = np.mean(pnls)
    return out, mean_daily, hit_rate, hits

def hit_rate(res_df, df, price_col="Close", allow_short=True, tau: float = 0):
    """
    Calcula el hit rate del modelo, o sea: qué porcentaje de veces la dirección que predijo 
    el modelo coincide con la dirección real del precio entre un día y el siguiente.  
    Básicamente mira si el trade hubiese ganado $1 según la señal que hubiese tomado.

    Inputs
    ------
    res_df : pd.DataFrame
        DataFrame de resultados con y_true y y_pred para cada día de test.
    df : pd.DataFrame
        DataFrame original con precios históricos (sirve para obtener el precio del día anterior).
    price_col : str
        Nombre de la columna de precio a usar (por defecto "Close").
    allow_short : bool
        Si es True, el modelo puede apostar en corto. Si es False, solo toma trades largos.
    tau : float
        Umbral mínimo de retorno predicho para decidir si operar o no (para filtrar ruido).

    Output
    ------
    hit_rate : float
        Proporción de aciertos (entre 0 y 1) según si la señal hubiese generado ganancia positiva.
    """

    test_days = res_df.index
    prices = df[price_col]
    hits = []
    for day in test_days:
        loc = prices.index.get_loc(day)
        prev_idx = loc - 1
        
        # precios real previo real (t-1), predicho (t), real (t)
        p_prev = float(prices.iloc[prev_idx])             # P_{d-1}
        p_pred = float(res_df.at[day, "y_pred"])          # \hat P_d
        p_true = float(res_df.at[day, "y_true"])          # P_d (real)

        # retornos real y predicho (sobre P_{d-1})
        r_real = (p_true - p_prev) / p_prev
        r_pred = (p_pred - p_prev) / p_prev

        if allow_short:
            s = 1.0 if r_pred >  tau else (-1.0 if r_pred < -tau else 0.0)
        else:
            s = 1.0 if r_pred >  tau else 0.0

        pnl = s * r_real  # ganancia/pérdida por $1 ese día
        if pnl > 0:
            hits.append(1)
        else:
            hits.append(0)

    hit_rate = np.sum(hits)/len(hits)
    return hit_rate

