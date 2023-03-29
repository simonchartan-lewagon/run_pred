import pandas as pd

def clean_data(path):
    """
    Takes the raw .csv dataset and cleans it to make it
    ready for preprocessing and feature engineering.
    """
    # Importing the raw dataset
    run = pd.read_csv(
        path,
        sep =';'
    )

    # Dataset initial variable names and units
    ## - athlete
    ## - gender
    ## - timestamp
    ## - distance (m)
    ## - elapsed time (s)
    ## - elevation gain (m)
    ## - average heart rate (bpm)

    # Renaming the columns with convenient names
    run.columns = [
        'athlete_id',
        'gender',
        'timestamp',
        'distance',
        'time',
        'elevation_gain',
        'average_heart_rate'
    ]

    # Formatting the columns
    ## Categorical variables
    for col in ['athlete_id', 'gender']:
        run[col] = run[col].astype('object')

    ## Datetime variables
    run.timestamp = pd.to_datetime(run.timestamp)

    ## Numerical variables
    for col in ['distance','time','elevation_gain','average_heart_rate']:
        run[col] = run[col].astype('float')


    # Drop duplicates
    run = run.drop_duplicates()

    # Cleaning columns

    ## athlete_id
    ### --> Nothing to do

    ## gender
    ### --> Dropping the 2 athletes that have no gender (no imputation possible)
    run = run.loc[~((run.athlete_id == 15655281) | (run.athlete_id == 46817575))]

    ## timestamp
    ### --> Removing athlete_id & timestamp duplicates
    run = run.sort_values(
        ['athlete_id','timestamp','average_heart_rate', 'distance'],
        ascending = [True, True, True, False]).reset_index(drop = True)
    run = run.drop_duplicates(subset = ['athlete_id','timestamp'])

    ## distance
    ### --> Removing left and right outliers, i.e. including 2km to 45 km courses
    run = run[run.distance <= 45000]
    run = run[(run.distance >= 2000)]

    ## time
    ### --> Removing left and right outliers, i.e. including 10 minutes to ~6 hours courses
    run = run[run.time <= 20000]
    run = run[run.time >= 600]

    ## elevation_gain
    ### --> Removing right outliers, i.e including only <2000m of elevation_gain courses
    run = run[(run.elevation_gain <= 2000)]

    ## average_heart_rate
    ### --> Removing left and right outliers, i.e. including only 100 bpm to 200 bpm courses
    run = run[((run.average_heart_rate >= 100) & (run.average_heart_rate <= 200)) | (run.average_heart_rate.isna())]

    ### --> Dropping the NaN rows for average_heart_rate
    ### We might come back to this point and decide to rather drop the whole column
    ### in case the model performance does not increase with the average_heart_rate
    ### and its derived features.
    run = run[~(run.average_heart_rate.isna())]

    ## pace
    ### --> Removing the pace outliers
    run['pseudo_distance'] = run.distance + 10 * run.elevation_gain # 1
    run['pseudo_pace'] = (run.time/60) / (run.pseudo_distance/1000) # min/km
    run = run[((run.pseudo_pace >= 2.5) & (run.pseudo_pace <= 15))]

    # Finally, reindexing the dataset and re-formatting
    run = run.reset_index(drop = True)
    run = run[[
        'time',
        'distance',
        'elevation_gain',
        'average_heart_rate',
        'gender',
        'timestamp'
    ]]

    return run


if __name__ == '__main__' :
    dataset = clean_data('raw_data/raw-data-kaggle.csv')
    print(dataset.shape)
    print(dataset.head())
    assert(dataset.shape == (22036,6))
