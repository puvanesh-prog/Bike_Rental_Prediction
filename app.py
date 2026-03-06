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
st.set_page_config(page_title="Bike Rental Analytics Pro", page_icon="🚲", layout="wide")

# Paths for Streamlit Cloud
DATA_PATH = "hour.csv" # Hourly data is more granular and better for prediction
SAVE_DIR = "bike_model_files"
MODEL_PATH = os.path.join(SAVE_DIR, "bike_hour_model.joblib")
PROCESSED_DATA_PATH = os.path.join(SAVE_DIR, "processed_bike_data.csv")

# --- 2. MODEL TRAINING (HOURLY BASIS) ---
def train_bike_model():
    if not os.path.exists(DATA_PATH):
        st.error(f"File '{DATA_PATH}' not found! Please upload hour.csv to GitHub.")
        st.stop()
        
    with st.spinner("Analyzing Hourly Patterns & Training Model..."):
        df = pd.read_csv(DATA_PATH)
        
        # Target is 'cnt' (Total rentals)
        # Dropping casual and registered as they sum up to cnt
        X = df.drop(['cnt', 'casual', 'registered', 'dteday', 'instant'], axis=1)
        y = df['cnt']
        
        num_features = X.select_dtypes(include=['float64', 'int64']).columns
        cat_features = X.select_dtypes(include=['object']).columns
        
        preprocessor = ColumnTransformer([
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
        ])
        
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ])
        
        model.fit(X, y)
        
        # Save assets
        if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
        joblib.dump(model, MODEL_PATH)
        
        df['Predicted_Count'] = model.predict(X)
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        st.success("Hourly Prediction Engine Ready!")

if not os.path.exists(MODEL_PATH):
    train_bike_model()

@st.cache_resource
def load_data():
    return pd.read_csv(PROCESSED_DATA_PATH), joblib.load(MODEL_PATH)

df, model = load_data()

# --- 3. UI NAVIGATION ---
st.sidebar.title("🚲 Rental Intelligence")
page = st.sidebar.radio("Navigate:", ["🏠 Dashboard", "🕒 Hourly Trends", "📈 Accuracy", "🔮 Demand Predictor", "💡 Business Strategy"])

# --- PAGE 1: DASHBOARD ---
if page == "🏠 Dashboard":
    st.title("🚲 Urban Bike Sharing Analytics")
    st.image("https://images.unsplash.com/photo-1558591710-4b4a1ae0f04d?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Avg Rentals/Hr", int(df['cnt'].mean()))
    col3.metric("Peak Demand", df['cnt'].max())

    st.markdown("""
    ### Why Two Datasets?
    - **Day.csv**: Used for long-term seasonal trends.
    - **Hour.csv**: (Used here) Provides granular data including specific hours of the day, allowing for precise demand forecasting.
    """)

# --- PAGE 2: HOURLY TRENDS ---
elif page == "🕒 Hourly Trends":
    st.title("🕒 Daily & Hourly Demand Patterns")
    
    tab1, tab2 = st.tabs(["Hourly Heatmap", "Weather Impact"])
    with tab1:
        hourly_map = df.groupby('hr')['cnt'].mean().reset_index()
        st.plotly_chart(px.line(hourly_map, x='hr', y='cnt', title="Average Rentals by Hour of Day", markers=True), use_container_width=True)
        st.info("Notice the peaks during office commute hours (8 AM and 5 PM).")
        

    with tab2:
        st.plotly_chart(px.scatter(df.sample(1000), x="temp", y="cnt", color="hum", title="Temperature & Humidity vs Demand"), use_container_width=True)

# --- PAGE 3: ACCURACY ---
elif page == "📈 Accuracy":
    st.title("📈 Model Performance Metrics")
    st.plotly_chart(px.scatter(df.sample(500), x='cnt', y='Predicted_Count', trendline="ols", title="Actual vs Predicted"), use_container_width=True)
    
    st.subheader("Top Predictors")
    importances = model.named_steps['regressor'].feature_importances_
    feats = model.named_steps['preprocessor'].get_feature_names_out()
    feat_df = pd.DataFrame({'Feature': feats, 'Importance': importances}).sort_values('Importance', ascending=False).head(10)
    st.plotly_chart(px.bar(feat_df, x='Importance', y='Feature', orientation='h'), use_container_width=True)

# --- PAGE 4: DEMAND PREDICTOR ---
elif page == "🔮 Demand Predictor":
    st.title("🔮 Real-Time Demand Forecasting")
    
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            hr = st.slider("Hour of Day", 0, 23, 9)
            temp = st.slider("Temperature (Normalized)", 0.0, 1.0, 0.5)
            hum = st.slider("Humidity", 0.0, 1.0, 0.5)
        with c2:
            season = st.selectbox("Season", [1, 2, 3, 4], format_func=lambda x: ["Spring", "Summer", "Fall", "Winter"][x-1])
            workingday = st.radio("Working Day?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            weathersit = st.selectbox("Weather Condition", [1, 2, 3, 4], format_func=lambda x: ["Clear", "Mist", "Light Snow", "Heavy Rain"][x-1])
        with c3:
            mnth = st.slider("Month", 1, 12, 6)
            holiday = st.radio("Holiday?", [0, 1])
            windspeed = st.slider("Windspeed", 0.0, 1.0, 0.1)
            
        submit = st.form_submit_button("Predict Demand")
        
    if submit:
        # Create input dataframe based on model features
        input_data = pd.DataFrame({
            'season': [season], 'yr': [1], 'mnth': [mnth], 'hr': [hr], 'holiday': [holiday],
            'weekday': [1], 'workingday': [workingday], 'weathersit': [weathersit],
            'temp': [temp], 'atemp': [temp], 'hum': [hum], 'windspeed': [windspeed]
        })
        
        prediction = model.predict(input_data)[0]
        st.metric("Estimated Bike Demand", f"{int(prediction)} Bikes")
        

# --- PAGE 5: STRATEGY ---
elif page == "💡 Business Strategy":
    st.title("💡 Strategic Fleet Management")
    st.markdown("""
    ### Core Insights:
    1. **Rush Hour Readiness**: Demand spikes at **8 AM** and **5-6 PM**. Fleet should be concentrated in residential and business districts during these windows.
    2. **Weather Sensitivity**: Rentals drop by over **60%** during 'Weather Situation 3/4' (Rain/Snow). Use this time for bike maintenance.
    3. **The 'Working Day' Factor**: Registered users dominate working days, while casual users peak on weekends. Market 'Weekend Passes' specifically to casual riders.
    """)
