from fastapi import FastAPI
from run_pred.interface.main import predict_race_times

api = FastAPI()

@api.get('/')
async def index():
    return {'Statut': 'OK'}

@api.get('/prediction')
async def predict(
    distance_km: float,
    elevation_gain_m: int,
    average_heart_rate: int,
    timestamp: str,
    gender: str):

    X_pred_dict = {
        'distance': distance_km * 1000,
        'elevation_gain': elevation_gain_m,
        'average_heart_rate': average_heart_rate,
        'timestamp': timestamp,
        'gender': gender
        }

    results_dict = predict_race_times(X_pred_dict)
    return results_dict

# http://127.0.0.1:8000/prediction?distance_km=10&elevation_gain_m=0&average_heart_rate=150&timestamp=2023-04-03%2019:30:00&gender=M
