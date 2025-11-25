import numpy as np
import pandas as pd
import pandas as pd
from utils.gnn import *
from mlxtend.frequent_patterns import apriori, association_rules

def create_gnn_embeddings(dfs, 
                             start, 
                             end, 
                             feat_cols,
                             tau_pear = 0.5,
                             tau_lift = 1.2,
                             alpha = 0.5):
    """
    Genera los embeddings GNN para un conjunto de acciones usando dos ideas: 
    (1) la correlación de Pearson, y (2) la co-dependencia de movimientos entre acciones (lift). 
    Después arma la matriz A_hat normalizada para poder usarla en un GNN, y crea la matriz X de features 
    alineada temporalmente para todos los stocks.  
    En resumen: arma el grafo, normaliza todo, y te deja listas las features propagadas.

    Inputs
    ------
    dfs : dict
        Diccionario {stock: dataframe} con precios históricos.
    start : str o Timestamp
        Fecha de inicio del rango de datos que quiero usar.
    end : str o Timestamp
        Fecha de fin.
    feat_cols : list
        Lista de features a usar para cada acción (por ejemplo ["Close", "Volume"]).
    tau_pear : float
        Umbral mínimo de correlación de Pearson para mantener una arista.
    tau_lift : float
        Umbral del lift para agregar co-movimientos fuertes.
    alpha : float
        Peso para combinar Pearson y Lift (alpha*Pearson + (1-alpha)*Lift).

    Outputs
    -------
    Z : np.ndarray
        Embeddings GNN ya propagados. Shape (T, N, F).
    dates : DatetimeIndex
        Fechas exactas que quedaron alineadas para todos los stocks.
    stocks : list
        Lista de los tickers en el mismo orden que aparece en Z.
    W_pear : np.ndarray
        Matriz de pesos basada solo en correlación de Pearson.
    W_lift : np.ndarray
        Matriz de pesos basada en lift (sin normalizar).

    """


    stocks = [stock for stock in dfs.keys()]
    # Combinar precios
    close = []
    for s, df in dfs.items():
        col = "Close"
        close.append(df[[col]].rename(columns={col: s})) # toma la columna de "Close" y le cambia el nombre al del "Stock"

    prices = pd.concat(close, axis=1).sort_index()
    prices = prices.asfreq("B").ffill()      # alineo a días hábiles y relleno
    prices = prices.loc[start:end]

    # Retornos semanales (más estables)
    returns = prices.pct_change().dropna()

    ######
    # 1) Matriz de pesos W = |rho| para la correlación de Pearson, sparsificada por umbral (sparsificar viene de generar varios ceros)
    ######

    corr_pearson = returns.corr(method="pearson").abs()  # |ρ_ij|

    W_pear = corr_pearson.abs().copy().values
    W_pear[W_pear < tau_pear] = 0.0
    np.fill_diagonal(W_pear, 0.0) # limpio la diagonal de la matriz pesos

    #####
    # 2) Matriz de pesos W para co dependencia de acciones
    #####

    R = returns.values.copy()
    T_ret, N_ret = R.shape    # T = cantidad de periodos, N = cantidad de stocks
    assert N_ret == len(stocks)  # checkeo que coincidan por las dudas

    # R es la matriz con los valores de retornos, la convierto en una matriz de signos
    signs = np.sign(R)

    # Soporte individual
    """ (signs != 0) es  una matriz buleana que se fija si hubo variacion o no en un día, osea si el retorno es mayor o menor que 0.
    Si la acción se movio -> 1
    Si la acción no se movio -> 0
    .sum Suma a lo largo del tiempo t (sentido vertical de la matriz), para cada acción
    Luego se divide por la cantidad de periodos T
    Probabilisticamente, esto significa: 
    support_ind ≈ P(stock i se mueve (sube o baja) en una semana dada) -> 'semana' porque se utilizan los retornos semanales 
    """
    support_ind = (signs != 0).sum(axis=0) / T_ret   # shape (N,)   -> vector de longitud N (cantidad de stocks)

    support_pair = np.zeros((N_ret, N_ret), dtype=float)   # Para cada par de acciones (i,j) obtengo el support
    for i in range(N_ret):
        for j in range(N_ret):
            same_dir = (signs[:, i] * signs[:, j] > 0)    # signs[:,i] es el vector de signos de la acción i a lo largo del tiempo. 
                                                          # si suben juntas 1, si van en distintas acciones -1. Los -1  se vuelven 0.  
            support_pair[i, j] = same_dir.sum() / T_ret   # Divido la cantidad de veces que dos acciones suben juntas por la cantidad de periodos de tiempo 
    
    eps = 1e-8
    lift = np.zeros((N_ret, N_ret), dtype=float)          # lift clásico entre dos evecntos A y B
    """
    lift(A,B) = P(A ∩ B) /  P(A)P(B)
    
    P(A) = support_ind i
    P(B) = support_ind j
    P(A ∩ B) = support_pair i,j
    
    """
    for i in range(N_ret):
        for j in range(N_ret):
            denom = support_ind[i] * support_ind[j] + eps
            lift[i, j] = support_pair[i, j] / denom

    # umbralizacion del lift
    W_lift = lift.copy()
    W_lift[W_lift < tau_lift] = 0.0

    # normalizacion de W_lift a [0,1]
    max_l = W_lift.max()   
    if max_l > 0:
        min_l = W_lift[W_lift > 0].min() if (W_lift > 0).any() else 0.0
        W_lift_norm = (W_lift - min_l) / (max_l - min_l + 1e-8)
        W_lift_norm[W_lift_norm < 0] = 0.0  # por seguridad
        W = alpha * W_pear + (1.0 - alpha) * W_lift_norm
    else:
        W = W_pear

    
    #####
    # 3) Combinacion Pearson + Apriori
    #####
    
    np.fill_diagonal(W, 0.0)  # sin self-loop en W (se agregan aparte)

    # 2) Self-loops
    I = np.eye(W.shape[0])
    A_monio = W + I

    # 3) Matriz D
    D = A_monio.sum(axis=1)
    D[D==0] = 1.0  # para evitar division por cero por si una fila es todo cero (mientras haya self-loop esto no ocurría de todas formas)
    D_inv_sqrt = 1.0/np.sqrt(D)
    D_inv_sqrt = np.diag(D_inv_sqrt)  # Paso de (N,) a (N,N) para poder hacer la multiplicación matricial

    # 4) A_hat
    A_hat = D_inv_sqrt @ A_monio @ D_inv_sqrt

    # 5) Matriz X (features)
    aligned = {}
    for s, df in dfs.items():
        df = df[feat_cols].sort_index()         # ordenar por fecha
        aligned[s] = df

    for s in stocks:
        aligned[s] = aligned[s].loc[start:end]

    # empezamos con las fechas del primer stock (todos los stocks tienen las mismas fechas)
    dates = aligned[stocks[0]].index
    for s in stocks[1:]:
        dates = dates.intersection(aligned[s].index)

    dates = dates.sort_values()

    T = len(dates)
    N = len(stocks)
    F = len(feat_cols)

    X = np.zeros((T, N, F), dtype=float)  # X[t, i, k] = feature k del stock i en el día t

    for i, s in enumerate(stocks):
        X[:, i, :] = aligned[s].loc[dates, feat_cols].values

    # Normalizo todos los stocks
    X_flat = X.reshape(-1, F)   # todos los días, todos los stocks

    min_vals = X_flat.min(axis=0)
    max_vals = X_flat.max(axis=0)

    X_norm = (X - min_vals) / (max_vals - min_vals + 1e-8)


    Z = np.zeros_like(X_norm)

    for t in range(T):
        # X[t] tiene shape (N, F)
        # A_hat tiene shape (N, N)
        # Resultado: (N, F)
        Z[t] = A_hat @ X_norm[t]

    return Z, dates, stocks, W_pear, W_lift