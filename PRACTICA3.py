import os
import time
import warnings
import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, freeze_support

# plotting opcional
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Modelos opcionales
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HW_AVAILABLE = True
except Exception:
    HW_AVAILABLE = False

# Parámetros (configurables)
FORECAST_HORIZON_WEEKS = 4
RESULTS_FILE = "forecast_resultados_mejorado.csv"
MIN_WEEKS = 12
MIN_TOTAL_SALES = 10
SERVICE_LEVEL_Z = 1.645  # z para ~95% service level
LEAD_TIME_WEEKS = 0.4285  # ~3 días

# ------------------ Métricas ------------------

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE seguro: evita división por cero. Devuelve pct."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float(np.mean(np.abs((y_true - y_pred))))
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE en %"""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom != 0
    if mask.sum() == 0:
        return float(np.mean(np.abs(y_true - y_pred)))
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.array(y_true, dtype=float) - np.array(y_pred, dtype=float))**2)))

# -------------- Inventario ----------------

def _calculate_inventory_params(train_sales: pd.Series, forecast_pred: np.ndarray) -> Dict[str, Any]:
    """Calcula Seguridad, Punto de Reorden y Cantidad a pedir."""
    sigma_demand = float(np.std(train_sales)) if len(train_sales) > 1 else 0.0
    SS = SERVICE_LEVEL_Z * sigma_demand * np.sqrt(LEAD_TIME_WEEKS)
    avg_forecast_per_week = float(np.mean(forecast_pred)) if forecast_pred.size > 0 else 0.0
    DLT = avg_forecast_per_week * LEAD_TIME_WEEKS
    ROP = DLT + SS
    last_stock = float(train_sales.iloc[-1]) if len(train_sales) > 0 else 0.0
    Q_reorder = max(0.0, ROP - last_stock)
    return {
        "Safety_Stock": int(round(max(0.0, SS))),
        "Reorder_Point": int(round(max(0.0, ROP))),
        "Qty_to_Order": int(round(max(0.0, Q_reorder))),
    }

# -------------- Procesamiento por SKU --------------

def process_sku(group) -> Dict[str, Any] | None:
    """Procesa una serie de ventas por SKU/StoreID y devuelve un dict de resultados."""
    (sku, store), series = group
    start_time = time.time()

    series = series.sort_values("InvoiceDate").copy()
    series = series.set_index("InvoiceDate")

    # Reindex semanal (W-SUN)
    series = series.resample("W-SUN").sum().fillna(0)

    # Filtros
    if len(series) < MIN_WEEKS:
        return None
    if series["Units_Sold"].sum() < MIN_TOTAL_SALES:
        return None

    # Winsorization
    pos = series["Units_Sold"][series["Units_Sold"] > 0]
    if len(pos) > 5:
        upper = pos.quantile(0.99)
        lower = pos.quantile(0.01)
        series["Units_Sold"] = np.clip(series["Units_Sold"], lower, upper)

    # Train/test split
    train = series[:-FORECAST_HORIZON_WEEKS].copy()
    test = series[-FORECAST_HORIZON_WEEKS:].copy()
    preds = []

    # Prophet
    try:
        if PROPHET_AVAILABLE and len(train) >= MIN_WEEKS:
            df_prophet = train[["Units_Sold"]].reset_index().rename(columns={"InvoiceDate":"ds","Units_Sold":"y"})
            df_prophet["y"] = np.log1p(df_prophet["y"].clip(lower=0.0))
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                        seasonality_mode='additive', changepoint_prior_scale=0.1)
            m.fit(df_prophet)
            future = m.make_future_dataframe(periods=FORECAST_HORIZON_WEEKS, freq="W-SUN")
            fc = m.predict(future)
            yhat = np.expm1(fc["yhat"].iloc[-FORECAST_HORIZON_WEEKS:].values)
            preds.append(np.maximum(yhat, 0.0))
    except Exception as e:
        logger.debug(f"Prophet failed for {sku}/{store}: {e}")

    # Holt-Winters
    try:
        if HW_AVAILABLE and len(train) >= 8:
            ts = np.log1p(train["Units_Sold"].astype(float).values)
            seasonal = 'add' if len(ts) >= 52 else None
            sp = 52 if len(ts) >= 52 else None
            hw = ExponentialSmoothing(ts, trend='add', seasonal=seasonal, seasonal_periods=sp)
            hw_fit = hw.fit(optimized=True)
            hw_fc = np.expm1(hw_fit.forecast(FORECAST_HORIZON_WEEKS))
            preds.append(np.maximum(hw_fc, 0.0))
    except Exception as e:
        logger.debug(f"Holt-Winters failed for {sku}/{store}: {e}")

    # Promedio móvil
    try:
        ma = train["Units_Sold"].tail(4).mean() if len(train) >= 4 else train["Units_Sold"].mean()
        preds.append(np.full(FORECAST_HORIZON_WEEKS, max(0.0, ma)))
    except Exception as e:
        logger.debug(f"Rolling mean failed for {sku}/{store}: {e}")

    # Ensemble
    stacked = np.vstack(preds)
    ensemble = np.median(stacked, axis=0)
    hist_avg = max(1.0, train["Units_Sold"].mean())
    upper_cap = max(10 * hist_avg, ensemble.max())
    ensemble = np.clip(ensemble, 0.0, upper_cap)

    # Métricas
    y_true = test["Units_Sold"].astype(float).values
    mape_val = safe_mape(y_true, ensemble)

    # Inventario
    inv = _calculate_inventory_params(train["Units_Sold"], ensemble)

    elapsed = time.time() - start_time

    return {
        "SKU": sku,
        "Store": store,
        "MAPE": round(mape_val, 3),
        "Forecast": ensemble.tolist(),
        "Test": y_true.tolist(),
        "Safety_Stock": inv["Safety_Stock"],
        "Reorder_Point": inv["Reorder_Point"],
        "Qty_to_Order": inv["Qty_to_Order"],
        "Runtime_sec": round(elapsed, 2)
    }

# ------------------ Ejecución principal ------------------

if __name__ == "__main__":
    freeze_support()

    # Cargar datos
    df = pd.read_csv("ventas.csv", parse_dates=["InvoiceDate"])

    # Renombrar columnas a las esperadas
    df = df.rename(columns={
        "StockCode": "Product_ID",
        "Country": "Store_ID",
        "Quantity": "Units_Sold"
    })

    # Convertir tipos y limpiar
    df["Units_Sold"] = pd.to_numeric(df["Units_Sold"], errors="coerce").fillna(0)
    df = df[df["Units_Sold"] >= 0]

    # Agrupar por producto y tienda
    groups = list(df.groupby(["Product_ID", "Store_ID"]))

    logger.info(f"Procesando {len(groups)} series...")
    results = []

    with Pool(max(1, cpu_count() - 1)) as pool:
        for r in tqdm(pool.imap_unordered(process_sku, groups), total=len(groups)):
            if r is not None:
                results.append(r)

    # Guardar resultados
    out_df = pd.DataFrame(results)
    out_df.to_csv(RESULTS_FILE, index=False)
    logger.info(f"Guardado {RESULTS_FILE} con {len(out_df)} SKUs procesados.")
