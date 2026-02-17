import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class DataLoader:

    def generate_synthetic_data(self, num_skus=20, num_days=365):
        np.random.seed(42)

        startDate = datetime(2023, 1, 1)
        records = []

        for sku in range(1, num_skus + 1):
            baseDemand = np.random.randint(20, 50)
            price = np.random.uniform(500, 2000)

            for day in range(num_days):
                currentDate = startDate + timedelta(days=day)

                # Weekly seasonality
                dayOfWeek = currentDate.weekday()
                seasonalFactor = 1.3 if dayOfWeek >= 5 else 1.0

                # Promotion randomly
                promotionFlag = np.random.choice([0, 1], p=[0.9, 0.1])

                promoBoost = 1.5 if promotionFlag == 1 else 1.0

                demand = int(baseDemand * seasonalFactor * promoBoost + np.random.normal(0, 5))

                inventoryLevel = np.random.randint(10, 100)

                unitsSold = min(demand, inventoryLevel)

                records.append([
                    currentDate,
                    f"SKU_{sku}",
                    f"Category_{(sku % 5) + 1}",
                    unitsSold,
                    price,
                    promotionFlag,
                    inventoryLevel
                ])

        df = pd.DataFrame(records, columns=[
            "date",
            "sku_id",
            "category",
            "units_sold",
            "price",
            "promotion_flag",
            "inventory_level"
        ])

        return df



if __name__ == "__main__":
    loader = DataLoader()
    df = loader.generate_synthetic_data()

    baseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    savePath = os.path.join(baseDir, "data", "raw")

    os.makedirs(savePath, exist_ok=True)

    filePath = os.path.join(savePath, "sales_data.csv")
    df.to_csv(filePath, index=False)
