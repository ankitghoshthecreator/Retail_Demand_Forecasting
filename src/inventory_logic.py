import pandas as pd
import os
import numpy as np


class InventoryLogic:

    def __init__(self, dataPath):
        self.dataPath = dataPath

    def load_data(self):
        df = pd.read_csv(self.dataPath, parse_dates=["date"])
        return df

    def compute_demand_stats(self, df):

        demandStats = (
            df.groupby("sku_id")["units_sold"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "avg_demand", "std": "demand_std"})
        )

        return demandStats

    def compute_replenishment(self, df):

        latestInventory = (
            df.sort_values("date")
            .groupby("sku_id")
            .tail(1)[["sku_id", "inventory_level"]]
        )

        demandStats = self.compute_demand_stats(df)

        merged = latestInventory.merge(demandStats, on="sku_id")

        leadTime = 7
        Z = 1.65

        merged["safety_stock"] = (
            Z * merged["demand_std"] * np.sqrt(leadTime)
        )

        merged["reorder_point"] = (
            merged["avg_demand"] * leadTime + merged["safety_stock"]
        )

        merged["reorder_quantity"] = np.where(
            merged["inventory_level"] < merged["reorder_point"],
            (merged["avg_demand"] * (leadTime + 7)) - merged["inventory_level"],
            0
        )

        return merged


if __name__ == "__main__":

    baseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataPath = os.path.join(baseDir, "data", "processed", "featured_data.csv")

    inv = InventoryLogic(dataPath)
    df = inv.load_data()
    result = inv.compute_replenishment(df)

    print(result.head())