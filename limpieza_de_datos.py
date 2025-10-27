# notebook: optimizacion_inventario.ipynb

# 1. Importar librerías
import pandas as pd
import numpy as np
from datasets import load_dataset
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 2. Cargar dataset
dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K", split="train")

# 3. Seleccionar un SKU / tienda para trabajar
df = dataset.to_pandas()
# Ejemplo: filtrar product_id = X, store_id = Y
sku = df[(df["product_id"] == 0) & (df["store_id"] == 0)].copy()

# 4. Preparación de datos
# Convertir fecha
sku["dt"] = pd.to_datetime(sku["dt"])
# Agregar ventas diarias si está en horas
sku_daily = sku.resample("D", on="dt").agg({"sale_amount": "sum"}).reset_index()
sku_daily = sku_daily.rename(columns={"dt": "ds", "sale_amount": "y"})

# Crear variables de lag
sku_daily["lag1"] = sku_daily["y"].shift(1)
sku_daily["lag7"] = sku_daily["y"].shift(7)
sku_daily = sku_daily.dropna()

# 5. División entrenamiento y prueba
train = sku_daily.iloc[:-30]
test = sku_daily.iloc[-30:]

# 6. Modelado con Prophet
m = Prophet()
m.fit(train[["ds","y"]])
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

# 7. Evaluación
preds = forecast.set_index("ds")["yhat"].loc[test["ds"]]
y_true = test.set_index("ds")["y"]
mae = mean_absolute_error(y_true, preds)
rmse = np.sqrt(mean_squared_error(y_true, preds))
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# 8. Visualización
m.plot(forecast)
plt.show()
