import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBRegressor

# ---------- CONFIG & PATHS ----------

def get_paths():
    """Setup absolute paths robustly for deployment."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    processed_dir = os.path.join(data_dir, "processed")
    
    os.makedirs(processed_dir, exist_ok=True)
    
    return {
        "processed": os.path.join(processed_dir, "featured_data.csv"),
        "raw": os.path.join(data_dir, "raw", "sales_data.csv")
    }

PATHS = get_paths()

# ---------- CORE ENGINE (CACHED IN MEMORY) ----------

@st.cache_data(show_spinner="Preparing Retail Data...")
def get_processed_data():
    """Generate and process data, caching the dataframe."""
    from src.data_loader import DataLoader
    from src.feature_engineering import FeatureEngineering

    # 1. Generate Raw Data if missing
    if not os.path.exists(PATHS["raw"]):
        loader = DataLoader()
        df_raw = loader.generate_synthetic_data()
        os.makedirs(os.path.dirname(PATHS["raw"]), exist_ok=True)
        df_raw.to_csv(PATHS["raw"], index=False)

    # 2. Extract Features
    fe = FeatureEngineering(PATHS["raw"])
    df_featured = fe.process()
    return df_featured

@st.cache_resource(show_spinner="Training Forecast Model...")
def get_trained_model(df):
    """Train the XGBoost model and cache the object in memory (No disk dependency)."""
    feature_cols = [
        "day_of_week", "month", "week_of_year", "lag_1", "lag_7",
        "rolling_mean_7", "price", "promotion_flag", "inventory_level", "stockout_flag"
    ]
    
    X = df[feature_cols]
    y = df["units_sold"]

    model = XGBRegressor(
        n_estimators=100,  # Slightly lower for faster cloud startup
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X, y)
    return model

# ---------- BUSINESS LOGIC ----------

def calculate_abc_analysis(df):
    df["revenue"] = df["units_sold"] * df["price"]
    sku_revenue = df.groupby("sku_id")["revenue"].sum().reset_index().sort_values(by="revenue", ascending=False)
    total_revenue = sku_revenue["revenue"].sum()
    sku_revenue["cumulative_share"] = sku_revenue["revenue"].cumsum() / total_revenue
    
    def classify(x):
        if x <= 0.7: return "A"
        if x <= 0.9: return "B"
        return "C"
        
    sku_revenue["abc_category"] = sku_revenue["cumulative_share"].apply(classify)
    return sku_revenue

def get_replenishment_recommendation(sku_data):
    latest_inventory = sku_data.sort_values("date").tail(1)["inventory_level"].values[0]
    avg_demand = sku_data["units_sold"].mean()
    demand_std = sku_data["units_sold"].std()
    
    lead_time = 7 
    z_score = 1.65 
    
    safety_stock = z_score * demand_std * np.sqrt(lead_time)
    reorder_point = avg_demand * lead_time + safety_stock
    
    reorder_qty = max(0, (avg_demand * (lead_time + 7)) - latest_inventory) if latest_inventory < reorder_point else 0
        
    return {
        "current_inventory": latest_inventory,
        "reorder_point": reorder_point,
        "recommended_qty": reorder_qty,
        "safety_stock": safety_stock
    }

# ---------- DASHBOARD UI ----------

def main():
    # 1. Load Data & Model
    df = get_processed_data()
    model = get_trained_model(df)
    
    # 2. Global Predictions
    feature_cols = [
        "day_of_week", "month", "week_of_year", "lag_1", "lag_7",
        "rolling_mean_7", "price", "promotion_flag", "inventory_level", "stockout_flag"
    ]
    df["predicted_demand"] = model.predict(df[feature_cols])

    # 3. Sidebar
    st.sidebar.title("SRR-DFS Control Panel")
    st.sidebar.markdown("---")
    
    category_list = ["All"] + sorted(df["category"].unique().tolist())
    selected_category = st.sidebar.selectbox("Filter by Category", category_list)
    filtered_df = df[df["category"] == selected_category] if selected_category != "All" else df
    
    sku_list = sorted(filtered_df["sku_id"].unique().tolist())
    selected_sku = st.sidebar.selectbox("Select Target SKU", sku_list)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Refresh All Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    # 4. Main Area
    st.markdown("# Smart Retail Replenishment & Demand Forecasting")
    
    sku_data = df[df["sku_id"] == selected_sku]
    abc_table = calculate_abc_analysis(df)
    sku_abc = abc_table[abc_table["sku_id"] == selected_sku]["abc_category"].values[0]
    rec = get_replenishment_recommendation(sku_data)
    
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ABC Class", sku_abc)
    m2.metric("Avg Demand", f"{sku_data['units_sold'].mean():.1f}")
    m3.metric("Current Stock", f"{rec['current_inventory']:.0f}")
    status = "REORDER" if rec['current_inventory'] < rec['reorder_point'] else "HEALTHY"
    m4.metric("Status", status, delta="-Needs Stock" if status=="REORDER" else "OK", delta_color="inverse")

    # Tabs
    t1, t2, t3 = st.tabs(["📊 Forecast", "📦 Inventory", "📈 ABC Analysis"])
    
    with t1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sku_data["date"], y=sku_data["units_sold"], name="Actual", line=dict(color='#00CC96')))
        fig.add_trace(go.Scatter(x=sku_data["date"], y=sku_data["predicted_demand"], name="Forecast", line=dict(dash='dash', color='#EF553B')))
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.write("### Recommended Reorder")
        if rec['recommended_qty'] > 0:
            st.warning(f"Order **{rec['recommended_qty']:.0f}** units.")
        else:
            st.success("Levels are healthy.")
            
        stock_fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = rec['current_inventory'],
            title = {'text': "Stock vs Reorder Point"},
            gauge = {
                'axis': {'range': [0, max(rec['current_inventory'], rec['reorder_point']) * 1.5]},
                'bar': {'color': "white"},
                'steps': [{'range': [0, rec['reorder_point']], 'color': "red"}],
                'threshold': {'line': {'color': "yellow", 'width': 4}, 'value': rec['reorder_point']}
            }
        ))
        stock_fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(stock_fig, use_container_width=True)

    with t3:
        fig_abc = px.pie(abc_table, names='abc_category', values='revenue', title='Revenue Mix')
        fig_abc.update_layout(template="plotly_dark")
        st.plotly_chart(fig_abc, use_container_width=True)

if __name__ == "__main__":
    main()