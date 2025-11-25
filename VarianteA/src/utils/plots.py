import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
from utils.gnn import *
from utils.metrics import * 

def show_results_forecasting(
        res):
    """
    Muestra en un solo gráfico cómo rindió el modelo durante los días de test:
    curva real, curva predicha, puntos donde acertó la dirección y puntos donde falló.
    También calcula y muestra métricas clave como MSE, retorno acumulado y hit rate.

    Inputs
    ------
    res : tuple
        Resultado completo que devuelve la función de forecasting:
            - res[0]: DataFrame con y_true, y_pred, y_true_norm, y_pred_norm.
            - res[1]: diccionario con parámetros del experimento (incluye el df original).

    Que hace
    -----------
    - Calcula MSE (en escala normalizada).
    - Calcula retorno diario simulado por dólar (PnL) y el retorno acumulado.
    - Calcula el hit rate real.
    - Grafica: precio real, precio predicho, precio del día anterior (para ver dirección),
    y marca con puntos verdes/rojos dónde el modelo acertó o erró.

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

    # PLOT mejorado
    plt.figure(figsize=(10,5))
    plt.plot(res_df.index, y_true, color="#1f77b4", lw=2.2, label="Close real")          # azul suave
    plt.plot(res_df.index, y_pred, color="#ff7f0e", lw=2.2, label="Close predicho")      # naranja
    plt.plot(res_df.index, y_prev, "--", color="#2ca02c", lw=1.5, alpha=0.7, label="Real (t-1)")  # verde punteada

    # Marcadores sobre la curva predicha
    plt.scatter(res_df.index[hit],  y_pred[hit],  color="limegreen", edgecolor="black", s=60, zorder=4, label="Dirección OK")
    plt.scatter(res_df.index[~hit], y_pred[~hit], color="red",                          s=60, marker="x", lw=2, zorder=4, label="Dirección FAIL")

    plt.title(f"MSE = {mse:.5f}\nRetorno acumulado en {data['horizon_days']} días = {retorno_acumulado:.2%}\nGanancia media diaria por $: {mean_daily:.2%}\nAciertos de dirección: {hit_rate:.1f}%", fontsize=12)
    plt.xlabel("Fecha"); plt.ylabel("Precio de cierre")
    plt.legend(frameon=True, facecolor="white", edgecolor="gray")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

def show_mse_vs_volume(
        res,
        mse_on: str = "norm",   # "norm" (default) usa y_*_norm; "raw" usa y_* reales
        smooth: int | None = None  # p.ej. 3, 5 o 7 para media móvil del MSE
    ):
    """
    Superpone MSE diario vs Volumen real por día usando el resultado `res`.
    - Eje izquierdo: MSE diario (por punto).
    - Eje derecho: Volumen real (desde df).

    Parámetros
    ----------
    res : tuple
        res[0] = res_df con columnas: y_true, y_pred, y_true_norm, y_pred_norm
        res[1] = dict con al menos data["df"] (serie original, con 'Volume')
    mse_on : {"norm", "raw"}
        - "norm": MSE calculado como (y_pred_norm - y_true_norm)^2 (recomendado).
        - "raw":  MSE calculado como (y_pred - y_true)^2.
    smooth : int | None
        Tamaño de ventana para media móvil del MSE. None para sin suavizado.
    """
    res_df = res[0]
    data = res[1]
    df = data["df"]

    # Elegir columnas según mse_on
    if mse_on == "norm":
        if not {"y_true_norm", "y_pred_norm"}.issubset(res_df.columns):
            raise ValueError("Faltan columnas 'y_true_norm'/'y_pred_norm' en res_df.")
        err = (res_df["y_pred_norm"] - res_df["y_true_norm"])**2
        mse_label = "MSE diario (norm)"
    elif mse_on == "raw":
        if not {"y_true", "y_pred"}.issubset(res_df.columns):
            raise ValueError("Faltan columnas 'y_true'/'y_pred' en res_df.")
        err = (res_df["y_pred"] - res_df["y_true"])**2
        mse_label = "MSE diario (raw)"
    else:
        raise ValueError("mse_on debe ser 'norm' o 'raw'.")

    # Suavizado opcional
    if smooth and smooth > 1:
        err_plot = err.rolling(smooth, min_periods=1).mean()
        mse_label += f" (MA{smooth})"
    else:
        err_plot = err

    # Localizar la columna de Volumen (tolerante a nombres)
    vol_candidates = [c for c in df.columns if c.lower() in ("volume", "vol")]
    if not vol_candidates:
        raise ValueError("No se encontró columna de volumen en df (ej: 'Volume' o 'Vol').")
    vol_col = vol_candidates[0]

    # Alinear el volumen al mismo índice del período de test (res_df.index)
    vol = df.loc[res_df.index, vol_col].astype(float)

    # Correlación simple (Pearson) entre MSE diario y Volumen
    # (dropna por si hubiera faltantes)
    aligned = pd.concat([err, vol], axis=1, keys=["mse", "vol"]).dropna()
    corr = aligned["mse"].corr(aligned["vol"])

    # Plot
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()

    # Volumen como barras (eje derecho)
    ax2.bar(res_df.index, vol, alpha=0.25, width=1.0, label="Volumen", edgecolor="none")

    # MSE como línea (eje izquierdo)
    ax1.plot(res_df.index, err_plot, lw=2.0, label=mse_label)

    ax1.set_xlabel("Fecha")
    ax1.set_ylabel("MSE por día")
    ax2.set_ylabel(f"Volumen ({vol_col})")

    title_corr = f"Correlación Pearson(MSE, Vol) = {corr:.3f}" if np.isfinite(corr) else "Correlación no disponible"
    ax1.set_title(f"MSE diario vs Volumen — {title_corr}")

    # Formateo del volumen (miles, millones, etc.)
    def human_readable(x, _):
        # formato corto
        abs_x = abs(x)
        if abs_x >= 1e9:  return f"{x/1e9:.1f}B"
        if abs_x >= 1e6:  return f"{x/1e6:.1f}M"
        if abs_x >= 1e3:  return f"{x/1e3:.1f}K"
        return f"{x:.0f}"
    ax2.yaxis.set_major_formatter(FuncFormatter(human_readable))

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=True)

    ax1.grid(True, alpha=0.25)
    fig.tight_layout()
    plt.show()


def grid_mse_and_hit(results):
    """
    Genera dos heatmaps (uno de MSE y otro de hit rate) para comparar varios modelos 
    sobre varios stocks al mismo tiempo. Es básicamente una vista rápida para ver 
    qué modelo rinde mejor por acción y en promedio.

    Inputs
    ------
    results : dict
        Diccionario anidado con la forma:
            {
                "modelo_1": {
                    "AAPL": (df_res, {"mse": ..., "hit_rate": ...}),
                    "MSFT": (...),
                    ...
                },
                "modelo_2": { ... },
                ...
            }
        Cada entrada tiene las métricas ya calculadas para ese modelo y ese stock.

    Que hace
    -----------
    - Extrae los MSE y hit rates para cada (modelo, stock).
    - Arma dos matrices (grid) con esos valores.
    - Calcula promedios por modelo.
    - Plotea dos heatmaps:
        1) MSE por modelo y por stock.
        2) Hit rate (%) por modelo y por stock.
    - Dentro de cada celda escribe el valor numérico para que quede más claro.

    Outputs
    -------
    No devuelve nada. Muestra los dos gráficos.
    """

    models = [model for model in results.keys()]

    stocks = [stock for stock in results[models[0]].keys()]

    mse_grid = np.zeros((len(models), len(stocks)), dtype=float)
    hit_grid = np.zeros((len(models), len(stocks)), dtype=float)

    for i, model in enumerate(models):
        for j, stock in enumerate(stocks):
            res = results[model][stock]
            mse_grid[i, j] = res[1]["mse"]
            hit_grid[i, j] = res[1]["hit_rate"]

    # Promedios por modelo
    mse_avg = mse_grid.mean(axis=1)       # shape (num_models,)
    hit_avg = hit_grid.mean(axis=1)

    # Figura
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), constrained_layout=True)

    # 1) HEATMAP de MSE
    ax = axes[0]
    im = ax.imshow(mse_grid, aspect="auto", cmap="coolwarm")

    title_mse = "MSE por modelo y stock | "
    title_mse += " • ".join([f"{models[i]} avg = {mse_avg[i]:.4f}" for i in range(len(models))])
    ax.set_title(title_mse, fontsize=13)

    ax.set_xticks(np.arange(len(stocks)))
    ax.set_xticklabels(stocks, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)
    ax.set_ylabel("Modelo")

    # Valores dentro de cada celda
    for i in range(len(models)):
        for j in range(len(stocks)):
            ax.text(j, i, f"{mse_grid[i, j]:.4f}",
                    ha="center", va="center", color="white")

    # Barra
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("MSE")

    # 2) HEATMAP de HIT RATE
    ax2 = axes[1]
    im2 = ax2.imshow(hit_grid, aspect="auto")

    title_hit = "Hit Rate (%) por modelo y stock | "
    title_hit += " • ".join([f"{models[i]} avg = {hit_avg[i]:.1f}%" for i in range(len(models))])
    ax2.set_title(title_hit, fontsize=13)

    ax2.set_xticks(np.arange(len(stocks)))
    ax2.set_xticklabels(stocks, rotation=45, ha="right")
    ax2.set_yticks(np.arange(len(models)))
    ax2.set_yticklabels(models)
    ax2.set_xlabel("Stock")
    ax2.set_ylabel("Modelo")

    # Valores dentro de cada celda
    for i in range(len(models)):
        for j in range(len(stocks)):
            ax2.text(j, i, f"{hit_grid[i, j]:.1f}",
                    ha="center", va="center", color="white")

    # Barra
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label("Hit Rate (%)")

    plt.show()
