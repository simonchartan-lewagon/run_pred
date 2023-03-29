def heart_rate_category(run):
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
    #création d'un intervalle
    interval = [run.average_heart_rate.mean()-(run.average_heart_rate.std()/2), run.average_heart_rate.mean()+(run.average_heart_rate.std()/2)]

    #catégorison selon cet intervalle
    run['heart_rate_category_1'] = (run['average_heart_rate'] < interval[0])*1
    run['heart_rate_category_2'] = ((run['average_heart_rate'] > interval[0]) & (run['average_heart_rate'] < interval[1]))*1
    run['heart_rate_category_3'] = (run['average_heart_rate'] > interval[1])*1

    return run



def categorize_heart_rate_by_user_max(run):
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

    # grouper les données par identifiant de coureur et trouver la valeur maximale de la colonne "average_heart_rate"
    max_heart_rates = run.groupby('athlete_id')['average_heart_rate'].max()

    # créer une nouvelle colonne avec la fréquence cardiaque maximale associée à chaque identifiant de coureur
    run['max_heart_rate'] = run['athlete_id'].map(max_heart_rates)

    # Catégorisé les fréquence cardiaque moyenne par rapport à la  fréquence cardiaque moyenne max de l'utilisateur
    run['heart_rate_low'] = (run['average_heart_rate'] < run.max_heart_rate*0.75)*1
    run['heart_rate_medium'] = ((run['average_heart_rate'] > run.max_heart_rate*0.75) & (run['average_heart_rate'] < run.max_heart_rate*0.85))*1
    run['heart_rate_hight'] = (run['average_heart_rate'] >  run.max_heart_rate*0.85)*1

    return run
