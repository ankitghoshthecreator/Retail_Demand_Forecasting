import pandas as pd
import os
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


class ForecastingModel:

    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.model = None

    def load_data(self):
        df = pd.read_csv(self.dataPath, parse_dates=["date"])
        return df

    def train_test_split(self, df):

        splitDate = df["date"].quantile(0.8)

        train = df[df["date"] <= splitDate]
        test = df[df["date"] > splitDate]

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

        X_train = train[featureCols]
        y_train = train["units_sold"]

        X_test = test[featureCols]
        y_test = test["units_sold"]

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):

        self.model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):

        predictions = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        print(f"RMSE: {rmse:.2f}")

        return rmse

    def save_model(self, savePath):
        joblib.dump(self.model, savePath)


if __name__ == "__main__":

    baseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataPath = os.path.join(baseDir, "data", "processed", "featured_data.csv")
    modelDir = os.path.join(baseDir, "models")

    os.makedirs(modelDir, exist_ok=True)

    fm = ForecastingModel(dataPath)

    df = fm.load_data()
    X_train, X_test, y_train, y_test = fm.train_test_split(df)

    fm.train_model(X_train, y_train)
    fm.evaluate(X_test, y_test)

    modelPath = os.path.join(modelDir, "xgb_model.pkl")
    fm.save_model(modelPath)

    print("Model training completed and saved.")