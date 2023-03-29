import numpy as np
import pandas as pd

def feature_engineering(run) :

    #convert timestamp into datetime
    run.timestamp = pd.to_datetime(run.timestamp)

    #add new rows of engineered features
    run['pace'] = pace(run['distance'], run['time'], run['elevation_gain'])
    run['average_speed'] = (run['distance']/run['time'])*3.6
    run['elevation_gain_per_km'] = run['elevation_gain']/(run['distance']/1000)
    run['sin_day'], run['cos_day'] = weekday(run['timestamp'])
    run['sin_month'], run['cos_month'] = month(run['timestamp'])
    run['AM'], run['PM'] = hour_1(run['timestamp'])
    run['dawn'], run['morning'], run['noon'], run['afternoon'], run['evening'] = hour_2(run['timestamp'])
    run['winter'], run['spring'], run['summer'], run['autumn'] = season(run['timestamp'])

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

def race_category(run):
    """
    Computes the race category of a running activity based on a combination of the distance covered, elevation gain,
    time spent running, and average heart rate. The resulting race category is assigned to the input DataFrame `run`
    as three binary columns: `race_category_1`, `race_category_2`, and `race_category_3`.
    """
    run['new_metric'] = ((run['distance']+(run['elevation_gain']*10))/(run['time']*run['average_heart_rate']))*100

    interval = [run.new_metric.mean()-(run.new_metric.std()/2), run.new_metric.mean()+(run.new_metric.std()/2)]

    run['race_category_1'] = (run.new_metric < interval[0])*1
    run['race_category_2'] = ((run.new_metric > interval[0]) & (run.new_metric < interval[1]))*1
    run['race_category_3'] = (run.new_metric > interval[1])*1

    return run

def elevation_category(run):
    """
    Computes the elevation ratio of the runs and categorizes them into different
    categories. The computed columns are:
    elevation_m_per_km: shows how many meters of elevation are gained each kilometer.
    elevation_category_1: runs with no elevation or a small elevation (=rolling elevation). (<9.5 m/km)
    elevation_category_2: runs with a moderate/steep elevation. (9.5-29 m/km)
    elevation_category_3: runs with a very steep elevation. (29-48 m/km)
    elevation_category_4: runs with a mountainous elevation. (>48 m/km)
    """

    run['elevation_m_per_km'] = (run['elevation_gain'])/(run['distance']/1000)

    run['elevation_category_1'] = (run.elevation_m_per_km < 9.5)*1
    run['elevation_category_2'] = ((run.elevation_m_per_km < 29 )&(run.elevation_m_per_km <= 9.5))*1
    run['elevation_category_3'] = ((run.elevation_m_per_km < 48 )&(run.elevation_m_per_km <= 29))*1
    run['elevation_category_4'] = (run.elevation_m_per_km > 48)*1

    return run