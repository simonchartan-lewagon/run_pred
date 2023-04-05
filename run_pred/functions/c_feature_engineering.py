import numpy as np
import pandas as pd
from run_pred.functions.a_cleaning import clean_data
from run_pred.functions.b_splitting import split_data

def engineer_features(run_features: pd.DataFrame, run_target: pd.DataFrame = None) -> pd.DataFrame :
    """
    This function takes a clean dataset and creates all relevant features
    meant to add predictive power to the model.

    Args:
        run_features (pd.DataFrame): Clean dataset containing run features.
        run_target (pd.DataFrame, optional): Clean dataset containing run target.
            Default to None, indicating the function is in prediction mode.

    Returns:
        pd.DataFrame: Clean dataset with added features that may improve predictive power.
    """
    run_features = run_features.copy()

    # converting timestamp into datetime
    run_features.timestamp = pd.to_datetime(run_features.timestamp)

    # adding relevant features
    run_features['elevation_gain_per_km'] = run_features['elevation_gain']/(run_features['distance']/1000)
    run_features['sin_day'], run_features['cos_day'] = weekday(run_features['timestamp'])
    run_features['sin_month'], run_features['cos_month'] = month(run_features['timestamp'])
    run_features['day_am'], run_features['day_pm'] = hour_1(run_features['timestamp'])
    run_features['day_dawn'], run_features['day_morning'], run_features['day_noon'], run_features['day_afternoon'], run_features['day_evening'] = hour_2(run_features['timestamp'])
    run_features['season_winter'], run_features['season_spring'], run_features['season_summer'], run_features['season_autumn'] = season(run_features['timestamp'])

    # modify the behaviour depending on whether we are in prediction mode or train/test mode
    if run_target is None: # we are in prediction mode, we build 3 rows, 1 for each race_category
        add_tbl = pd.DataFrame({
            'race_category_1' : [1,0,0],
            'race_category_2' : [0,1,0],
            'race_category_3' : [0,0,1]
            })
        run_features.loc[1] = run_features.loc[0]
        run_features.loc[2] = run_features.loc[0]
        run_features = run_features.join(add_tbl)
    else: # we are in train/test mode
        run_features = race_category(run_features,run_target)

    run_features = elevation_category(run_features)

    # The 2 functions below are not executed as it does not add predictive power to the model.
    #run_features = heart_rate_category(run_features)
    #run_features = categorize_heart_rate_by_user_max(run_features)

    #drop the timestamp column
    run_features = run_features.drop(columns = ['timestamp'])

    return run_features


# creating cyclical features for weekdays
def weekday(timestamp: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    This function creates cyclical features for the weekday from the given timestamp series.

    Args:
        timestamp (pd.Series): A pandas series containing datetime values.

    Returns:
        Tuple of two pandas series (sin_day, cos_day) containing the sine and cosine transformed weekday
        values, respectively. The returned features have been transformed to represent cyclic
        features that allow modeling cyclical phenomena.
    """

    # creating a pd.Series Monday -> Sunday
    weekday = timestamp.dt.day_name()

    # creating weekday dictionary
    dict = {'Monday' : 1,
            'Tuesday' : 2,
            'Wednesday' : 3,
            'Thursday' : 4,
            'Friday' : 5,
            'Saturday': 6,
            'Sunday' : 7}

    # converting day into number
    weekday = weekday.map(dict)

    # creating cyclical features
    days_in_a_week = 7

    sin_day = np.sin(2*np.pi*(weekday-1)/days_in_a_week)
    cos_day = np.cos(2*np.pi*(weekday-1)/days_in_a_week)

    return sin_day, cos_day


# creating a cyclical feature for months
def month(timestamp: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    This function creates cyclical features for the month from the given timestamp series.

    Args:
        timestamp (pd.Series): A pandas series containing datetime values.

    Returns:
        Tuple of two pandas series (sin_month, cos_month) containing the sine and cosine transformed month
        values, respectively. The returned features have been transformed to represent cyclic
        features that allow modeling cyclical phenomena.
    """

    month = timestamp.dt.month
    months_in_a_year = 12

    # creating cyclical features
    sin_month = np.sin(2*np.pi*(month-1)/months_in_a_year)
    cos_month = np.cos(2*np.pi*(month-1)/months_in_a_year)

    return sin_month, cos_month

# creating a AM/PM split for hours
def hour_1(timestamp: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    This function creates binary features for AM and PM based on the given timestamp series.

    Args:
        timestamp (pd.Series): A pandas series containing datetime values.

    Returns:
        Tuple of two pandas series (hour_am, hour_pm) containing binary values 1 or 0
        based on whether the timestamp is before noon (AM) or after noon (PM), respectively.
    """

    hour = timestamp.dt.hour

    # creating am & pm categories
    hour_am = (hour<=12)*1
    hour_pm = (hour>12)*1

    return hour_am, hour_pm


# creating a dawn/morning/noon/afternoon/evening split for hours
def hour_2(timestamp: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    This function takes a pd.Series of timestamps and creates cyclical features for the hour of the day,
    representing whether it is in the dawn, morning, noon, afternoon or evening periods.

    Args:
    - timestamp (pd.Series): A pandas Series object containing timestamps for each row of data.

    Returns:
    A tuple containing five pandas Series objects representing the cyclical features for the dawn, morning,
    noon, afternoon and evening periods of the day.
    """

    hour = timestamp.dt.hour

    # creating relevant columns
    dawn = (hour<=7)*1
    morning = ((hour>7) & (hour <=10 ))*1
    noon = ((hour>10) & (hour <=13 ))*1
    afternoon = ((hour>13) & (hour <=16))*1
    evening = (hour>16)*1

    return dawn, morning, noon, afternoon, evening

def season(timestamp: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    This function takes a pd.Series object representing a timestamp column and creates 4 new
    columns to represent the season of the year in which the run took place.

    Parameters:
    timestamp (pd.Series): A pandas series containing datetime values representing the time of the run.

    Returns:
    tuple: A tuple of four pd.Series objects. The first represents whether the run took place in winter,
    the second represents whether it took place in spring, the third represents whether it took place in
    summer, and the fourth represents whether it took place in autumn.
    """
    month = timestamp.dt.month

    # creating relevant columns
    winter = (month<= 3)*1
    spring = ((month>3) & (month <= 6 ))*1
    summer = ((month>6) & (month <=9 ))*1
    autumn = (month>9) *1

    return winter, spring, summer, autumn


def race_category(run_features: pd.DataFrame, run_target: pd.Series) -> pd.DataFrame:
    """
    This function computes the race category of a running activity based on a combination of the distance covered, elevation gain,
    time spent running, and average heart rate. The resulting race category is assigned to the input DataFrame `run`
    as three binary columns: `race_category_1`, `race_category_2`, and `race_category_3`.

    Parameters:
    -----------
    run_features : pd.DataFrame
        A pandas DataFrame containing the features of running activities.
    run_target : pd.Series
        A pandas Series containing the target value of running activities.

    Returns:
    --------
    pd.DataFrame
        A pandas DataFrame with additional columns `race_category_1`, `race_category_2`, and `race_category_3`
        indicating the race category of the corresponding running activities.
    """

    run_features['race_category_metric'] = ((run_features['distance']+(run_features['elevation_gain']*10))/(run_target*run_features['average_heart_rate']))*100

    interval = [run_features.race_category_metric.mean()-(run_features.race_category_metric.std()/2), run_features.race_category_metric.mean()+(run_features.race_category_metric.std()/2)]

    run_features['race_category_1'] = (run_features.race_category_metric < interval[0])*1
    run_features['race_category_2'] = ((run_features.race_category_metric >= interval[0]) & (run_features.race_category_metric < interval[1]))*1
    run_features['race_category_3'] = (run_features.race_category_metric >= interval[1])*1

    run_features = run_features.drop(columns = ['race_category_metric'])

    return run_features

def elevation_category(run: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the elevation ratio of the runs and categorizes them into different
    categories. The computed columns are:
    elevation_m_per_km: shows how many meters of elevation are gained each kilometer.
    elevation_category_1: runs with no elevation or a small elevation (=rolling elevation). (<9.5 m/km)
    elevation_category_2: runs with a moderate/steep elevation. (9.5-29 m/km)
    elevation_category_3: runs with a very steep elevation. (>29 m/km)
    """

    run['elevation_m_per_km'] = (run['elevation_gain'])/(run['distance']/1000)

    run['elevation_category_1'] = (run.elevation_m_per_km < 9.5)*1
    run['elevation_category_2'] = ((run.elevation_m_per_km >= 9.5) & (run.elevation_m_per_km < 29))*1
    run['elevation_category_3'] = (run.elevation_m_per_km >= 29) *1

    run = run.drop(columns = ['elevation_m_per_km'])

    return run

def heart_rate_category(run: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes as input a 'run' dataframe containing information about a run, including average heart rate.
    It categorizes the average heart rate to identify the type of running performed by the runner.
    She then creates three columns in the dataframe to indicate the heart rate category of the run:
    'heart_rate_category_1' for the lowest category, 'heart_rate_category_2' for the intermediate category and 'heart_rate_category_3' for the highest category.
    The categories are determined by creating an interval centered on the average heart rate and classifying the average heart rate values ​​according to this interval.

    Arguments:
    - run: a dataframe containing information about a run, including average heart rate.

    Returns:
    - run: the updated dataframe with heart rate category columns added.
    """
    # Creating the medium interval
    interval = [run.average_heart_rate.mean()-(run.average_heart_rate.std()/2), run.average_heart_rate.mean()+(run.average_heart_rate.std()/2)]

    # Categorizing based on this interval
    run['heart_rate_category_1'] = (run['average_heart_rate'] < interval[0])*1
    run['heart_rate_category_2'] = ((run['average_heart_rate'] >= interval[0]) & (run['average_heart_rate'] < interval[1]))*1
    run['heart_rate_category_3'] = (run['average_heart_rate'] >= interval[1])*1

    return run

def categorize_heart_rate_by_user_max(run: pd.DataFrame) -> pd.DataFrame:
    """
    Categorizes average heart rates for a given run dataframe by comparing them to the maximum heart rate
    for each athlete.

    Args:
        run (pandas.DataFrame): A pandas DataFrame with the following columns: athlete_id (int),
        average_heart_rate (float).

    Returns:
        pandas.DataFrame: The same pandas DataFrame as input, but with three additional columns:
        max_heart_rate (float), heart_rate_low (int), heart_rate_medium (int), and heart_rate_high (int).
        The heart_rate_low, heart_rate_medium, and heart_rate_high columns indicate whether the
        average heart rate for a given athlete during a given run is below 75%, between 75% and 85%, or
        above 85% of their maximum heart rate, respectively.
    """

    # Group data by athlete_id and find the maximum average_heart_rate
    max_heart_rates = run.groupby('athlete_id')['average_heart_rate'].max()
    run['max_average_heart_rate'] = run['athlete_id'].map(max_heart_rates)

    thresh_1 = 0.80
    thresh_2 = 0.85

    # Categorize the rows based on where the average_heart_rate lies compared to the athlete maximum average_heart_rate
    run['heart_rate_low'] = (run['average_heart_rate'] < run.max_average_heart_rate*thresh_1)*1
    run['heart_rate_medium'] = ((run['average_heart_rate'] >= run.max_average_heart_rate*thresh_1) & (run['average_heart_rate'] < run.max_average_heart_rate*thresh_2))*1
    run['heart_rate_hight'] = (run['average_heart_rate'] >= run.max_average_heart_rate*thresh_2)*1

    run = run.drop(columns = ['athlete_id','max_average_heart_rate'])

    return run

if __name__ == '__main__' :
    # train/test mode
    dataset = clean_data('raw_data/raw-data-kaggle.csv')
    X_train_raw, X_test, y_train_raw, y_test = split_data(dataset)
    X_train_feat = engineer_features(X_train_raw,y_train_raw)
    X_test_feat = engineer_features(X_test,y_test)
    print(X_train_feat.shape)
    print(X_train_feat.head())
    print(X_train_feat.columns)
    print(X_test_feat.shape)
    print(X_test_feat.head())
    print(X_test_feat.columns)

    # prediction mode
    ## X_pred_raw = pd.DataFrame({
    ##     'distance': 10000,
    ##     'elevation_gain': 200,
    ##     'average_heart_rate': 150,
    ##     'timestamp': '2023-04-01 10:00:00',
    ##     'gender': 'M'
    ##     }, index = [0])
    ## X_pred_feat = engineer_features(X_pred_raw)
    ## print(X_pred_feat.head())
    ## print(X_pred_feat.shape)
    ## print(X_pred_feat.columns)
