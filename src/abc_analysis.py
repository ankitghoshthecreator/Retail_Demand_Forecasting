import pandas as pd
import os


class ABCAnalysis:

    def __init__(self, dataPath):
        self.dataPath = dataPath

    def load_data(self):
        df = pd.read_csv(self.dataPath)
        return df

    def compute_revenue(self, df):
        df["revenue"] = df["units_sold"] * df["price"]
        return df

    def aggregate_sku_revenue(self, df):
        skuRevenue = (
            df.groupby("sku_id")["revenue"]
            .sum()
            .reset_index()
            .sort_values(by="revenue", ascending=False)
        )
        return skuRevenue

    def classify_abc(self, skuRevenue):

        totalRevenue = skuRevenue["revenue"].sum()
        skuRevenue["cumulative_revenue"] = skuRevenue["revenue"].cumsum()
        skuRevenue["cumulative_percentage"] = (
            skuRevenue["cumulative_revenue"] / totalRevenue
        )

        def category(row):
            if row["cumulative_percentage"] <= 0.7:
                return "A"
            elif row["cumulative_percentage"] <= 0.9:
                return "B"
            else:
                return "C"

        skuRevenue["abc_category"] = skuRevenue.apply(category, axis=1)

        return skuRevenue


if __name__ == "__main__":

    baseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataPath = os.path.join(baseDir, "data", "processed", "featured_data.csv")

    abc = ABCAnalysis(dataPath)

    df = abc.load_data()
    df = abc.compute_revenue(df)
    skuRevenue = abc.aggregate_sku_revenue(df)
    skuRevenue = abc.classify_abc(skuRevenue)

    print(skuRevenue.head())