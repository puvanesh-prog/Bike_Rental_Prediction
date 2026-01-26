import requests

url = "https://bike-rental-pred.onrender.com/predict"

data = {
    "season": 1,
    "holiday": 0,
    "weekday": 3,
    "workingday": 1,
    "weathersit": 2,
    "temp": 0.5,
    "hum": 0.6,
    "windspeed": 0.15,
    "year": 1,
    "month": 7
}

print(requests.post(url, json=data).json())
