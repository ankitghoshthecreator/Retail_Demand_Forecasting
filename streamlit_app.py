import streamlit as st
import os
import sys

# Add the project root to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set page config for a premium look
st.set_page_config(
    page_title="SRR-DFS | Smart Retail Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import and run the main dashboard logic
from app.dashboard import main

if __name__ == "__main__":
    main()
