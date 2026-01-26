from flask import Flask, request, jsonify,send_from_directory
import joblib
import numpy as np
import pandas as pd

model = joblib.load("model/final_bike_model.pkl")  # stacked model
scaler = joblib.load("model/scaler.pkl")           # scaler

app = Flask(__name__, static_folder='static')


@app.route('/')
def homepage():
    return send_from_directory('static', 'bike_web.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # Feature Engineering
        df['windspeed_log'] = np.log1p(df['windspeed'])
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['temp_wind'] = df['temp'] * df['windspeed']
        df['summer_workingday'] = ((df['season'] == 3) & (df['workingday'] == 1)).astype(int)

        # Define model features
        model_features = [
            'season', 'holiday', 'weekday', 'workingday', 'weathersit',
            'temp', 'hum', 'windspeed', 'windspeed_log', 'year', 'month',
            'is_weekend', 'month_sin', 'month_cos', 'temp_wind', 'summer_workingday'
        ]

        df_model = df[model_features]

        # Columns scaled during training
        num_cols_final = [
            'temp','hum','windspeed','windspeed_log','year','month',
            'month_sin','month_cos','temp_wind','summer_workingday'
        ]

        # Scale  numeric columns
        df_model[num_cols_final] = scaler.transform(df_model[num_cols_final])


        # Prediction
        prediction = float(model.predict(df_model)[0])

        return jsonify({
            "input": data,
            "prediction_cnt": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
