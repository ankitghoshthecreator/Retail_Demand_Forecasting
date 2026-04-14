import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBRegressor
from datetime import datetime

# ---------- CONFIG & PATHS ----------

def get_paths():
    """Setup absolute paths robustly for deployment."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    processed_dir = os.path.join(data_dir, "processed")
    model_dir = os.path.join(base_dir, "models")
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    return {
        "processed": os.path.join(processed_dir, "featured_data.csv"),
        "model": os.path.join(model_dir, "xgb_model.pkl"),
        "raw": os.path.join(data_dir, "raw", "sales_data.csv")
    }

PATHS = get_paths()

# ---------- CACHED DATA PIPELINE ----------

@st.cache_data(show_spinner="Building Data Pipeline...")
def build_pipeline_cached():
    """Initialize data and train model if missing."""
    from src.data_loader import DataLoader
    from src.feature_engineering import FeatureEngineering

    # 1. Generate Data
    loader = DataLoader()
    df_raw = loader.generate_synthetic_data()
    
    raw_dir = os.path.dirname(PATHS["raw"])
    os.makedirs(raw_dir, exist_ok=True)
    df_raw.to_csv(PATHS["raw"], index=False)

    # 2. Feature Engineering
    fe = FeatureEngineering(PATHS["raw"])
    df_featured = fe.process()
    df_featured.to_csv(PATHS["processed"], index=False)

    # 3. Model Training
    feature_cols = [
        "day_of_week", "month", "week_of_year", "lag_1", "lag_7",
        "rolling_mean_7", "price", "promotion_flag", "inventory_level", "stockout_flag"
    ]
    
    X = df_featured[feature_cols]
    y = df_featured["units_sold"]

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y)
    model.save_model(PATHS["model"])
    
    return df_featured

@st.cache_data
def load_data():
    if not os.path.exists(PATHS["processed"]) or not os.path.exists(PATHS["model"]):
        return build_pipeline_cached()
    return pd.read_csv(PATHS["processed"], parse_dates=["date"])

@st.cache_resource
def load_model_cached():
    model = XGBRegressor()
    # Check if exists and is non-empty
    if not os.path.exists(PATHS["model"]) or os.path.getsize(PATHS["model"]) == 0:
        st.info("📦 Initializing model for the first time...")
        build_pipeline_cached()
    
    try:
        model.load_model(PATHS["model"])
    except Exception as e:
        st.warning(f"🔄 Model file corrupted, retraining: {e}")
        build_pipeline_cached()
        model.load_model(PATHS["model"])
        
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
    
    lead_time = 7  # Days
    z_score = 1.65 # 95% service level
    
    safety_stock = z_score * demand_std * np.sqrt(lead_time)
    reorder_point = avg_demand * lead_time + safety_stock
    
    reorder_qty = 0
    if latest_inventory < reorder_point:
        reorder_qty = (avg_demand * (lead_time + 7)) - latest_inventory
        
    return {
        "current_inventory": latest_inventory,
        "reorder_point": reorder_point,
        "recommended_qty": max(0, reorder_qty),
        "safety_stock": safety_stock
    }

# ---------- DASHBOARD UI ----------

def main():
    # Load assets
    df = load_data()
    model = load_model_cached()
    
    # Feature columns for prediction
    feature_cols = [
        "day_of_week", "month", "week_of_year", "lag_1", "lag_7",
        "rolling_mean_7", "price", "promotion_flag", "inventory_level", "stockout_flag"
    ]
    df["predicted_demand"] = model.predict(df[feature_cols])

    # Sidebar
    st.sidebar.title("SRR-DFS Control Panel")
    st.sidebar.markdown("---")
    
    category_list = ["All"] + sorted(df["category"].unique().tolist())
    selected_category = st.sidebar.selectbox("Filter by Category", category_list)
    
    if selected_category != "All":
        filtered_df = df[df["category"] == selected_category]
    else:
        filtered_df = df
        
    sku_list = sorted(filtered_df["sku_id"].unique().tolist())
    selected_sku = st.sidebar.selectbox("Select Target SKU", sku_list)
    
    st.sidebar.markdown("---")
    st.sidebar.info("**About SRR-DFS**\n\nThis system uses XGBoost to forecast demand and provides automated replenishment advice based on ABC classification and safety stock modeling.")
    
    if st.sidebar.button("Force Retrain Model"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    # Main Area
    st.markdown("# Smart Retail Replenishment & Demand Forecasting")
    
    # Top Level Metrics
    sku_data = df[df["sku_id"] == selected_sku]
    abc_table = calculate_abc_analysis(df)
    sku_abc = abc_table[abc_table["sku_id"] == selected_sku]["abc_category"].values[0]
    rec = get_replenishment_recommendation(sku_data)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("ABC Classification", sku_abc)
    with c2:
        st.metric("Avg Daily Demand", f"{sku_data['units_sold'].mean():.1f}")
    with c3:
        st.metric("Current Stock", f"{rec['current_inventory']:.0f}")
    with c4:
        status = "REORDER" if rec['current_inventory'] < rec['reorder_point'] else "HEALTHY"
        st.metric("Stock Status", status, delta="-Needs Restock" if status=="REORDER" else "OK", delta_color="inverse")

    # Tabs
    tab_forecast, tab_inventory, tab_abc = st.tabs(["📊 Demand Forecast", "📦 Inventory Logic", "📈 ABC Analysis"])
    
    with tab_forecast:
        st.subheader(f"Demand Trends & Forecast: {selected_sku}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sku_data["date"], y=sku_data["units_sold"], name="Actual Sales", line=dict(color='#00CC96')))
        fig.add_trace(go.Scatter(x=sku_data["date"], y=sku_data["predicted_demand"], name="ML Forecast", line=dict(dash='dash', color='#EF553B')))
        fig.update_layout(template="plotly_dark", height=500, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("View Raw Forecast Data"):
            st.dataframe(sku_data[["date", "units_sold", "predicted_demand", "price", "promotion_flag"]].tail(15))

    with tab_inventory:
        st.subheader("Replenishment Optimization Parameters")
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.write("### Model Parameters")
            st.write(f"**Lead Time:** 7 Days")
            st.write(f"**Confidence Level:** 95% (Z=1.65)")
            st.write(f"**Safety Stock calculated:** {rec['safety_stock']:.2f} units")
            st.write(f"**Reorder Point (ROP):** {rec['reorder_point']:.2f} units")
            
        with col_right:
            st.write("### Recommendation")
            if rec['recommended_qty'] > 0:
                st.warning(f"⚠️ Recommendation: Order **{rec['recommended_qty']:.0f}** units immediately.")
            else:
                st.success("✅ Current stock levels are sufficient.")
            
            # Stock level visualization
            stock_fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = rec['current_inventory'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Current Inventory vs ROP"},
                gauge = {
                    'axis': {'range': [0, max(rec['current_inventory'], rec['reorder_point']) * 1.5]},
                    'bar': {'color': "white"},
                    'steps': [
                        {'range': [0, rec['reorder_point']], 'color': "red"},
                        {'range': [rec['reorder_point'], 9999], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "yellow", 'width': 4},
                        'thickness': 0.75,
                        'value': rec['reorder_point']
                    }
                }
            ))
            stock_fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(stock_fig, use_container_width=True)

    with tab_abc:
        st.subheader("ABC Inventory Classification")
        st.write("Classification based on 70/20/10 revenue share distribution.")
        
        fig_abc = px.pie(abc_table, names='abc_category', values='revenue', 
                         title='Revenue Distribution by ABC Category',
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_abc.update_layout(template="plotly_dark")
        st.plotly_chart(fig_abc, use_container_width=True)
        
        st.write("### Pareto Analysis (Cumulative Revenue)")
        pareto_fig = px.line(abc_table.sort_values('cumulative_share'), 
                              x=range(len(abc_table)), y='cumulative_share',
                              labels={'x': 'Number of SKUs', 'cumulative_share': 'Cumulative Revenue %'})
        pareto_fig.add_hline(y=0.7, line_dash="dash", line_color="green", annotation_text="A (70%)")
        pareto_fig.add_hline(y=0.9, line_dash="dash", line_color="yellow", annotation_text="B (90%)")
        pareto_fig.update_layout(template="plotly_dark")
        st.plotly_chart(pareto_fig, use_container_width=True)

if __name__ == "__main__":
    main()