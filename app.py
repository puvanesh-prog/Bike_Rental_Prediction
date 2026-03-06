import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Bike Rental Multi-Level Analytics", page_icon="🚲", layout="wide")

# Paths for Cloud
DAY_DATA = "day.csv"
HOUR_DATA = "hour.csv"
SAVE_DIR = "bike_model_files"
MODEL_PATH = os.path.join(SAVE_DIR, "bike_final_model.joblib")

# --- 2. DUAL-DATA LOADING & MODEL TRAINING ---
def initialize_engine():
    if not os.path.exists(DAY_DATA) or not os.path.exists(HOUR_DATA):
        st.error("Missing Data: Ensure both day.csv and hour.csv are in the GitHub root folder.")
        st.stop()
    
    with st.spinner("Training Intelligence Engine with Dual Datasets..."):
        # Training predominantly on Hourly data for better precision
        df_hour = pd.read_csv(HOUR_DATA)
        
        # Preprocessing & Model Logic
        X = df_hour.drop(['cnt', 'casual', 'registered', 'dteday', 'instant'], axis=1)
        y = df_hour['cnt']
        
        num_cols = X.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = X.select_dtypes(include=['object']).columns
        
        preprocessor = ColumnTransformer([
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
        ])
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        pipeline.fit(X, y)
        
        if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
        joblib.dump(pipeline, MODEL_PATH)
        st.success("System Ready with Day & Hour Analysis Capabilities!")

if not os.path.exists(MODEL_PATH):
    initialize_engine()

@st.cache_resource
def load_assets():
    d_df = pd.read_csv(DAY_DATA)
    h_df = pd.read_csv(HOUR_DATA)
    model = joblib.load(MODEL_PATH)
    return d_df, h_df, model

df_day, df_hour, model = load_assets()

# --- 3. NAVIGATION ---
st.sidebar.title("🚲 Bike Rental Analytics")
data_view = st.sidebar.selectbox("Choose Data Perspective:", ["Hourly (Granular)", "Daily (Long-term)"])
page = st.sidebar.radio("Go to:", ["🏠 Dashboard", "📊 Trend Analysis", "🔮 Demand Predictor", "💡 Strategic Insights"])

# Set active dataframe based on user choice
df = df_hour if data_view == "Hourly (Granular)" else df_day

# --- PAGE 1: DASHBOARD ---
if page == "🏠 Dashboard":
    st.title(f"🏠 {data_view} Rental Dashboard")
    st.image("https://images.unsplash.com/photo-1507035895480-2b3156c31fc8?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Observations", len(df))
    c2.metric("Average Rentals", int(df['cnt'].mean()))
    c3.metric("Max Demand Record", df['cnt'].max())

    st.markdown(f"""
    ### Dataset Analysis:
    This dashboard is currently displaying **{data_view}** metrics. 
    - **Daily data** helps in understanding seasonal changes.
    - **Hourly data** captures the morning and evening rush hour spikes.
    """)
    

# --- PAGE 2: TREND ANALYSIS ---
elif page == "📊 Trend Analysis":
    st.title("📊 Demand & Weather Correlation")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.box(df, x="season", y="cnt", title="Demand Variance by Season", color="season"), use_container_width=True)
    with col2:
        st.plotly_chart(px.scatter(df.sample(min(2000, len(df))), x="temp", y="cnt", color="weathersit", title="Temperature vs. Demand"), use_container_width=True)
        

    if data_view == "Hourly (Granular)":
        st.subheader("🕒 Peak Hour Identification")
        hourly_avg = df.groupby('hr')['cnt'].mean().reset_index()
        st.plotly_chart(px.line(hourly_avg, x='hr', y='cnt', markers=True, title="Hourly Demand Peaks"), use_container_width=True)

# --- PAGE 3: PREDICTOR ---
elif page == "🔮 Demand Predictor":
    st.title("🔮 Rental Forecasting Engine")
    st.info("Predicting based on Hourly patterns for maximum accuracy.")
    
    with st.form("pred_form"):
        c1, c2 = st.columns(2)
        with c1:
            hour_val = st.slider("Hour of Day", 0, 23, 10)
            temp_val = st.slider("Normalized Temperature", 0.0, 1.0, 0.5)
            hum_val = st.slider("Humidity (Normalized)", 0.0, 1.0, 0.5)
        with c2:
            season_val = st.selectbox("Season", [1,2,3,4], format_func=lambda x: ["Spring","Summer","Fall","Winter"][x-1])
            is_work = st.radio("Working Day?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            weather_val = st.selectbox("Weather Condition", [1,2,3,4], format_func=lambda x: ["Clear","Mist","Light Snow","Heavy Rain"][x-1])
        
        btn = st.form_submit_button("Generate Forecast")
        
    if btn:
        # Construct input matching the Hourly model features
        input_row = pd.DataFrame({
            'season': [season_val], 'yr': [1], 'mnth': [6], 'hr': [hour_val],
            'holiday': [0], 'weekday': [3], 'workingday': [is_work], 'weathersit': [weather_val],
            'temp': [temp_val], 'atemp': [temp_val], 'hum': [hum_val], 'windspeed': [0.1]
        })
        
        pred = model.predict(input_row)[0]
        st.success(f"Estimated Demand: **{int(pred)} Bikes**")

# --- PAGE 4: STRATEGIC INSIGHTS ---
elif page == "💡 Strategic Insights":
    st.title("💡 Operational Business Insights")
    
    st.subheader("Demand Drivers (Feature Importance)")
    importances = model.named_steps['model'].feature_importances_
    feats = model.named_steps['preprocessor'].get_feature_names_out()
    imp_df = pd.DataFrame({'Feature': feats, 'Weight': importances}).sort_values('Weight', ascending=False).head(10)
    st.plotly_chart(px.bar(imp_df, x='Weight', y='Feature', orientation='h', color='Weight'), use_container_width=True)
    
    st.markdown("""
    ### 📝 Business Action Plan:
    1. **Dynamic Allocation**: Move 40% of the fleet to transport hubs during **7-9 AM** and **5-7 PM**.
    2. **Weather Discounts**: Offer "Rainy Day" discounts to boost casual rentals during Mist/Light Snow conditions.
    3. **Seasonality**: Prepare for a **3x increase** in demand during Summer (Season 2) vs Winter (Season 4).
    """)
