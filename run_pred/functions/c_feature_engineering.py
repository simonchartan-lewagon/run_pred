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
