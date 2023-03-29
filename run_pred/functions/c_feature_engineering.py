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
