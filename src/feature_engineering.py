import pandas as pd
import os


class FeatureEngineering:

    def __init__(self, dataPath):
        self.dataPath = dataPath

    def load_data(self):
        df = pd.read_csv(self.dataPath, parse_dates=["date"])
        return df

    def sort_data(self, df):
        df = df.sort_values(by=["sku_id", "date"])
        return df

    def create_time_features(self, df):
        df["day_of_week"] = df["date"].dt.weekday
        df["month"] = df["date"].dt.month
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        return df

    def create_lag_features(self, df):
        df["lag_1"] = df.groupby("sku_id")["units_sold"].shift(1)
        df["lag_7"] = df.groupby("sku_id")["units_sold"].shift(7)
        return df

    def create_rolling_features(self, df):
        df["rolling_mean_7"] = (
            df.groupby("sku_id")["units_sold"]
            .shift(1)
            .rolling(window=7)
            .mean()
        )
        return df

    def create_stockout_flag(self, df):
        df["stockout_flag"] = (
            (df["inventory_level"] == df["units_sold"]) &
            (df["inventory_level"] > 0)
        ).astype(int)
        return df

    def process(self):
        df = self.load_data()
        df = self.sort_data(df)
        df = self.create_time_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_stockout_flag(df)
        df = df.dropna()
        return df


if __name__ == "__main__":
    baseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rawPath = os.path.join(baseDir, "data", "raw", "sales_data.csv")
    processedDir = os.path.join(baseDir, "data", "processed")

    os.makedirs(processedDir, exist_ok=True)

    fe = FeatureEngineering(rawPath)
    df = fe.process()

    savePath = os.path.join(processedDir, "featured_data.csv")
    df.to_csv(savePath, index=False)

    print("Feature engineering completed.")
    print(df.head())