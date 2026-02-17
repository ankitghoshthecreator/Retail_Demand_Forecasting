# Smart Retail Replenishment & Demand Forecasting System (SRR-DFS)

## Overview
SRR-DFS is an intelligent retail analytics system designed to forecast SKU-level demand and generate automated inventory replenishment recommendations.

The system combines machine learning-based demand forecasting with ABC inventory classification and statistical safety stock modeling to minimize stockouts and reduce excess inventory.

---

## Problem Statement

Retail businesses often face:

- Stockouts leading to lost revenue
- Excess inventory increasing holding costs
- Poor SKU prioritization
- Manual replenishment decisions

SRR-DFS solves these issues using data-driven forecasting and automated reorder logic.

---

## Core Features

- SKU-level demand forecasting using XGBoost
- Weekly seasonality modeling (lag & rolling features)
- ABC inventory classification
- Safety stock calculation
- Automated reorder quantity recommendation
- Sell-through analysis
- Slow-moving stock alerts
- Streamlit dashboard for interactive insights

---

## System Architecture

1. Data Ingestion
2. Feature Engineering
3. Demand Forecasting Model
4. ABC Analysis
5. Inventory Optimization Logic
6. Dashboard Visualization

---

## Tech Stack

- Python 3.9
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Plotly
- Streamlit
- Supabase (PostgreSQL backend)

---

## Target Users

- Fashion Retailers
- Multi-SKU Retail Stores
- Inventory Managers
- Supply Chain Analysts

---

## Business Impact

- Reduced stockouts
- Improved inventory turnover
- Lower holding cost
- Better working capital utilization

---

## Project Structure

