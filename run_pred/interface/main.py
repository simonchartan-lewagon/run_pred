import pandas as pd
import joblib
import time
from run_pred.params import *
from run_pred.functions.c_feature_engineering import engineer_features
from run_pred.functions.f_training import load_model


test_dict = {
    'distance': 10000,
    'elevation_gain': 200,
    'average_heart_rate': 150,
    'timestamp': '2023-04-01 10:00:00',
    'gender': 'M'
}

def predict_race_times(X_pred_dict = test_dict):

    # transform the input data dictionary to a DataFrame
    X_pred_raw = pd.DataFrame(X_pred_dict, index = [0])

    # create the input data features
    X_pred_feat = engineer_features(X_pred_raw)

    # scale the data
    scaler_path = os.path.join(LOCAL_REGISTRY_PATH, "models", "StandardScaler.joblib")
    scaler = joblib.load(scaler_path)

    X_pred_feat[['distance', 'elevation_gain', 'average_heart_rate' , 'elevation_gain_per_km']] = pd.DataFrame(
        scaler.transform(X_pred_feat[['distance', 'elevation_gain', 'average_heart_rate' , 'elevation_gain_per_km']]),
        columns= ['distance', 'elevation_gain', 'average_heart_rate' , 'elevation_gain_per_km'])

    X_pred_feat_scaled = X_pred_feat

    # encode the data
    ohe_path = os.path.join(LOCAL_REGISTRY_PATH, "models", "OneHotEncoder.joblib")
    ohe = joblib.load(ohe_path)

    X_pred_feat_scaled.gender = pd.DataFrame(
        ohe.transform(X_pred_feat_scaled[['gender']]),
        columns = ohe.get_feature_names_out())

    X_pred_feat_scaled_encoded = pd.DataFrame(
        X_pred_feat_scaled,
        columns = ['sin_day', 'cos_day', 'sin_month', 'cos_month', 'day_am', 'day_pm',
       'day_dawn', 'day_morning', 'day_noon', 'day_afternoon', 'day_evening',
       'season_winter', 'season_spring', 'season_summer', 'season_autumn',
       'race_category_1', 'race_category_2', 'race_category_3',
       'elevation_category_1', 'elevation_category_2', 'elevation_category_3',
       'gender', 'distance', 'elevation_gain', 'average_heart_rate',
       'elevation_gain_per_km'])

    # load the model and predict
    model = load_model()
    results = model.predict(X_pred_feat_scaled_encoded)

    results_dict = {
       'race_category_1_pred_time' : round(results[0],0),
       'race_category_2_pred_time' : round(results[1],0),
       'race_category_3_pred_time' : round(results[2],0)
    }

    results_dict_display = {}

    for var in results_dict:
        value = results_dict.get(var)
        if value < 3600:
            results_dict_display[var] = time.strftime('%M min %S sec', time.gmtime(value))
        else:
            results_dict_display[var] = time.strftime('%H h %M min %S sec', time.gmtime(value))

    return results_dict_display

if __name__ == '__main__' :

    pred_dict = {
        'distance': 10000,
        'elevation_gain': 0,
        'average_heart_rate': 150,
        'timestamp': '2023-04-23 09:30:00',
        'gender': 'M'
        }

    print(predict_race_times(pred_dict))
