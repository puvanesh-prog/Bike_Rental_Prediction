# 🚲 Urban Bike Rental Demand Analytics & Forecasting Pro

An end-to-end Machine Learning solution to analyze and predict bike sharing demand using **Hourly** and **Daily** datasets. This project leverages **Random Forest Regression** to help urban planners and bike-share operators optimize fleet distribution.

## 🚀 Live Demo
**Interact with the live app here:** https://bit.ly/4s50O7q 

---

## 📌 Project Overview
Predicting bike rental demand is a complex challenge influenced by weather, time of day, and seasonal trends. This project provides a dual-layered analysis:
1. **Daily Insights**: Understanding long-term seasonal impacts.
2. **Hourly Precision**: Identifying specific rush-hour peaks for real-time fleet management.

### 🔑 Key Modules:
- **🏠 Dual-Data Dashboard**: Toggle between Daily and Hourly perspectives to view high-level metrics.
- **📊 Demand Heatmaps**: Visualizing peak usage times (Rush hours at 8 AM and 5 PM).
- **🔮 Real-Time Predictor**: A robust forecasting engine where users can input weather and time variables to get instant rental estimates.
- **💡 Operational Strategy**: Data-driven business recommendations for fleet optimization and maintenance scheduling.

---

## 🛠️ Tech Stack
- **Language**: Python 3.10+
- **Framework**: Streamlit
- **Machine Learning**: Scikit-Learn (Random Forest Regressor, Pipeline, ColumnTransformer)
- **Data Handling**: Pandas, Numpy
- **Visualization**: Plotly Express, Matplotlib
- **Deployment**: Streamlit Community Cloud

---

## 📂 Dataset Details
The project utilizes the UCI Machine Learning Repository's Bike Sharing Dataset:
- **hour.csv**: 17,379 records with hourly features (Temp, Humidity, Windspeed, Hour, etc.)
- **day.csv**: 731 records focusing on daily aggregates and seasonal patterns.

---

## ⚙️ Local Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/puvanesh-prog/Bike_Rental_Prediction.git](https://github.com/puvanesh-prog/Bike_Rental_Prediction.git)
