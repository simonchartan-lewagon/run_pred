from fastapi import FastAPI
from run_pred.interface.main import predict_race_times

api = FastAPI()

@api.get('/')
async def index() -> dict:
    """
    Returns a dictionary with a status message indicating that the function is working properly.

    Returns:
    - dict: A dictionary with a 'Statut' key and 'OK' value.
    """
    return {'Statut': 'OK'}

@api.get('/prediction')
async def predict(
    distance_km: float,
    elevation_gain_m: int,
    average_heart_rate: int,
    timestamp: str,
    gender: str) -> dict :

    """
    Predicts the race times for a given input and returns the results in a dictionary.

    Args:
    - distance_km (float): The distance of the race in kilometers.
    - elevation_gain_m (int): The total elevation gain of the race in meters.
    - average_heart_rate (int): The average heart rate of the participant during the race.
    - timestamp (str): The timestamp of the race in the format 'YYYY-MM-DD HH:MM:SS'.
    - gender (str): The gender of the participant ('Male' or 'Female').

    Returns:
    - dict: A dictionary with the predicted race times for each race category, in seconds.
        - 'race_category_1_pred_time' : the predicted time for race category 1
        - 'race_category_2_pred_time' : the predicted time for race category 2
        - 'race_category_3_pred_time' : the predicted time for race category 3
    """

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
