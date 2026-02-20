import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


# ---------- PATH SETUP ----------

baseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataDir = os.path.join(baseDir, "data")
processedDir = os.path.join(dataDir, "processed")
modelDir = os.path.join(baseDir, "models")

os.makedirs(processedDir, exist_ok=True)
os.makedirs(modelDir, exist_ok=True)

processedPath = os.path.join(processedDir, "featured_data.csv")
modelPath = os.path.join(modelDir, "xgb_model.pkl")


# ---------- AUTO BUILD PIPELINE IF MISSING ----------

def build_pipeline():

    from src.data_loader import DataLoader
    from src.feature_engineering import FeatureEngineering

    loader = DataLoader()
    df = loader.generate_synthetic_data()

    rawPath = os.path.join(dataDir, "raw")
    os.makedirs(rawPath, exist_ok=True)
    rawFile = os.path.join(rawPath, "sales_data.csv")
    df.to_csv(rawFile, index=False)

    fe = FeatureEngineering(rawFile)
    df = fe.process()
    df.to_csv(processedPath, index=False)

    featureCols = [
        "day_of_week",
        "month",
        "week_of_year",
        "lag_1",
        "lag_7",
        "rolling_mean_7",
        "price",
        "promotion_flag",
        "inventory_level",
        "stockout_flag"
    ]

    X = df[featureCols]
    y = df["units_sold"]

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X, y)
    joblib.dump(model, modelPath)


if not os.path.exists(processedPath) or not os.path.exists(modelPath):
    build_pipeline()


# ---------- LOAD DATA & MODEL ----------

df = pd.read_csv(processedPath, parse_dates=["date"])
model = joblib.load(modelPath)

featureCols = [
    "day_of_week",
    "month",
    "week_of_year",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "price",
    "promotion_flag",
    "inventory_level",
    "stockout_flag"
]

df["predicted_demand"] = model.predict(df[featureCols])


# ---------- UI ----------

st.title("Smart Retail Replenishment & Demand Forecasting System")

skuList = df["sku_id"].unique()
selectedSKU = st.selectbox("Select SKU", skuList)

skuData = df[df["sku_id"] == selectedSKU]

st.line_chart(skuData.set_index("date")[["units_sold", "predicted_demand"]])

df["revenue"] = df["units_sold"] * df["price"]

skuRevenue = (
    df.groupby("sku_id")["revenue"]
    .sum()
    .reset_index()
    .sort_values(by="revenue", ascending=False)
)

totalRevenue = skuRevenue["revenue"].sum()
skuRevenue["cumulative"] = skuRevenue["revenue"].cumsum() / totalRevenue

def abc_category(x):
    if x <= 0.7:
        return "A"
    elif x <= 0.9:
        return "B"
    else:
        return "C"

skuRevenue["abc_category"] = skuRevenue["cumulative"].apply(abc_category)

abcCategory = skuRevenue[skuRevenue["sku_id"] == selectedSKU]["abc_category"].values[0]

st.subheader(f"ABC Category: {abcCategory}")

latestInventory = skuData.sort_values("date").tail(1)["inventory_level"].values[0]
avgDemand = skuData["units_sold"].mean()
demandStd = skuData["units_sold"].std()

leadTime = 7
Z = 1.65

safetyStock = Z * demandStd * np.sqrt(leadTime)
reorderPoint = avgDemand * leadTime + safetyStock

if latestInventory < reorderPoint:
    reorderQty = (avgDemand * (leadTime + 7)) - latestInventory
else:
    reorderQty = 0

st.subheader("Replenishment Recommendation")
st.write(f"Current Inventory: {latestInventory:.2f}")
st.write(f"Reorder Point: {reorderPoint:.2f}")
st.write(f"Recommended Reorder Quantity: {reorderQty:.2f}")