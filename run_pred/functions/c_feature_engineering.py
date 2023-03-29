import numpy as np
import pandas as pd
from run_pred.functions.a_cleaning import clean_data
from run_pred.functions.b_splitting import split_data

def engineer_features(run) :

    """
    This function takes a clean dataset
    """

    #convert timestamp into datetime
    run.timestamp = pd.to_datetime(run.timestamp)

    #add engineered features
    #run['pace'] = pace(run['distance'], run['time'], run['elevation_gain'])
    #run['average_speed'] = (run['distance']/run['time'])*3.6
    run['elevation_gain_per_km'] = run['elevation_gain']/(run['distance']/1000)
    run['sin_day'], run['cos_day'] = weekday(run['timestamp'])
    run['sin_month'], run['cos_month'] = month(run['timestamp'])
    run['day_am'], run['day_pm'] = hour_1(run['timestamp'])
    run['day_dawn'], run['day_morning'], run['day_noon'], run['day_afternoon'], run['day_evening'] = hour_2(run['timestamp'])
    run['season_winter'], run['season_spring'], run['season_summer'], run['season_autumn'] = season(run['timestamp'])

    #drop the timestamp column
    run = run.drop(columns = ['timestamp'])

    return run


# creating a corriged pace function  (distance + elevation_gain):
def pace(distance, time, elevation_gain):

    adapted_distance = distance/1000 + ((elevation_gain//10)*0.1)
    pace = (time/60)/(adapted_distance)

    return round(pace, 2)

# creating a cyclical feature for weekdays

def weekday(timestamp):
    #creating rows Monday -> Saturday
    weekday = timestamp.dt.day_name()

    #weekday dictionary
    dict = {'Monday' : 1,
            'Tuesday' : 2,
            'Wednesday' : 3,
            'Thursday' : 4,
            'Friday' : 5,
            'Saturday': 6,
            'Sunday' : 7}

    #modify day into number
    weekday = weekday.map(dict)


    # Treat Cyclical Features
    days_in_a_week = 7

    sin_day = np.sin(2*np.pi*(weekday-1)/days_in_a_week)
    cos_day = np.cos(2*np.pi*(weekday-1)/days_in_a_week)

    return sin_day, cos_day


# creating a cyclical feature for months
def month(timestamp):
    #creating row 'month'
    month = timestamp.dt.month

    # Treat Cyclical Features
    months_in_a_year = 12

    sin_month = np.sin(2*np.pi*(month-1)/months_in_a_year)
    cos_month = np.cos(2*np.pi*(month-1)/months_in_a_year)

    return sin_month, cos_month

# creating a AM/PM split for hours
def hour_1(timestamp):
    #crating row 'hour'
    hour = timestamp.dt.hour

    #creating rows 'AM' & 'PM'
    hour_am = (hour<=12)*1
    hour_pm = (hour>12)*1

    return hour_am, hour_pm


# creating a dawn/morning/noon/afternoon/evening split for hours
def hour_2(timestamp):
    #crating row 'hour'
    hour = timestamp.dt.hour

    #création des colonnes dawn/morning/noon/afternoon/evening
    dawn = (hour<=8)*1
    morning = ((hour>8) & (hour <=11 ))*1
    noon = ((hour>11) & (hour <=14 ))*1
    afternoon = ((hour>14) & (hour <=17 ))*1
    evening = (hour>17)*1

    return dawn, morning, noon, afternoon, evening

def season(timestamp):
    #crating row 'hour'
    month = timestamp.dt.month

    #création des colonnes dawn/morning/noon/afternoon/evening
    winter = (month<= 3 )*1
    spring = ((month>3) & (month <= 6 ))*1
    summer = ((month>6) & (month <=9 ))*1
    autumn = (month>9) *1

    return winter, spring, summer, autumn




if __name__ == '__main__' :
    dataset = clean_data('raw_data/raw-data-kaggle.csv')
    X_train_raw, X_test, y_train_raw, y_test = split_data(dataset)
    X_train_feat = engineer_features(X_train_raw)
    print(X_train_feat.shape)
    print(X_train_feat.head())
