# Trail Running Race Predictor

Source dataset: https://www.kaggle.com/datasets/olegoaer/running-races-strava

The project aims to predict race times using a machine learning model trained on a dataset (after cleaning and rebalancing) of ~35000 races from 105 runners. There will be three different predictions based on the desired performance level or user level.

The initial dataset is composed of 42,116 rows and 7 columns, and includes features such as:
- athlete: the unique identifier for the athlete who completed the race
- gender: the binary category of the athlete, either Male (M) or Female (F)
- timestamp: the date and time (in minutes) when the race began
- distance (in meters): the distance of the race course
- elapsed time (in seconds): the time taken by the athlete to complete the race
- elevation gain (in meters): the elevation gain of the race course
- average heart rate (in BPM): the average heart rate of the athlete during the race

## Preprocessing

### Data cleaning

file: a_cleaning.py

The data has been cleaned by removing duplicates, searching and delate for outliers, and handling missing values.

### Data splitting

file: b_splitting.py

The dataset has been split in train/test set and features/target.

### Feature engineering

file: c_feature_engineering

Feature engineering has also been performed, including the creation of time-based features such as:
- creating a cyclical feature for weekdays: sin_day, cos_day

- creating a cyclical feature for months: sin_month, cos_month

- categorical features such as:
  - day_am, day_pm: categorize whether the race was held during the morning (am) or afternoon (pm) half of the day.
  - day_dawn, day_morning, day_noon, day_afternoon, day_evening: categorize the specific time of day when the race was held.
  - season_winter, season_spring, season_summer, season_autumn: categorize the season of the year when the race was held.

  elevation_category: feature computes the elevation ratio of the runs and categorizes them into different categories. The computation is done by dividing the elevation_gain by the distance and multiplying the result by 1000 to get the elevation gained per kilometer. Based on this, the runs are then categorized into the following categories:

    elevation_category_1: runs with no elevation or a small elevation gain (= rolling elevation) of less than 9.5 meters per kilometer.
    elevation_category_2: runs with a moderate/steep elevation gain of 9.5 to 29 meters per kilometer.
    elevation_category_3: runs with a very steep elevation gain of over 29 meters per kilometer.

The resulting categories are assigned to the input DataFrame run as binary columns: elevation_category_1, elevation_category_2, and elevation_category_3.

race_category: a feature that captures different performance levels based on a combination of distance covered, elevation gain, time spent running, and average heart rate.
The race_category feature is computed for each running activity and is assigned to the input DataFrame as three binary columns: race_category_1, race_category_2, and race_category_3. These columns indicate the level of performance associated with the running activity, with higher values corresponding to higher performance levels. The elevation_category feature categorizes the elevation of the running activity and provides additional information to the machine learning model to improve its predictions.

### Balancing

file: d_balancing.py

To balance the dataset by gender, the number of races performed by men and women has been equalized.

### Scaling/encoding

file: e_scaling_encoding.py

Data has been scaled using standard scaler for numerical values.
Additionally, the gender column has been encoded to numerical values using one-hot encoding.

### Final dataset
The final DataFrame consists 26 features, which are listed below:
- sin_day
- cos_day
- sin_month
- cos_month
- day_am
- day_pm
- day_dawn
- day_morning
- day_noon
- day_afternoon
- day_evening
- season_winter
- season_spring
- season_summer
- season_autumn
- race_category_1
- race_category_2
- race_category_3
- elevation_category_1
- elevation_category_2
- elevation_category_3
- gender
- distance
- elevation_gain
- average_heart_rate
- elevation_gain_per_km

## Training the ML model

file: f_training.py

Once the data is ready, we train the best model found during the R&D phase, which is a StackingRegressor composed of a RandomForest, GradientBoosting, XGB, and a LinearRegression as the final estimator.

The model achieved a high accuracy score of 96.9% in R² and a low mean absolute percentage error (MAPE) of 5.4%. This indicates that the model can effectively predict race times based on the input features.
N.B.: The model can be saved or loaded depending on needs.

Overall, the project demonstrates the effectiveness of machine learning in predicting race times and showcases the importance of feature engineering and data cleaning in achieving accurate predictions.


## Installation

To install the run_pred package and its required dependencies, you can follow the steps below:

First, ensure that you have Python 3.10 or later installed on your system. You can check this by opening a terminal or command prompt and running the following command:
```
python --version
```
If you don't have Python installed, you can download it from the official website: https://www.python.org/downloads/

Clone the repository or download the source code from the GitHub page:
```
git clone https://github.com/username/run_pred.git
```
Change into the project directory:
```
cd run_pred
```
Install the required dependencies by running the following command:
```
pip install -r requirements.txt
```
Once the dependencies have been installed, you can install the run_pred package itself by running the following command:
```
python setup.py install
```
That's it! You should now be able to use the run_pred package in your Python projects.

## Usage

The project utilizes a stacked machine learning model to predict race times based on various input features, including distance, date and time, average heart rate, and elevation gain. The model is capable of returning three different predicted race times based on the race_category of the event. Users can input the necessary data and the model will return the predicted race time for the corresponding race_category.

We have also created a web page where users can view a demo of the results of our work. The page is available at https://benito-p-run-pred-front-page-web-roqy7e.streamlit.app/.

## Contribution

We welcome contributions to this package, especially in the following areas:

- Improving the accuracy of the predictions for longer distances (e.g., marathon or ultra-marathon).
- Adapting the package to different datasets and use cases.
- Enhancing the visualization capabilities of the package.

If you're interested in contributing, please feel free to submit a pull request or open an issue to discuss your ideas. We appreciate your help in making this package more useful for the running community!

## Authors

- simonchartan-lewagon
- TomP81
- erkaminski
- benito-p

## Licence

MIT licence, see licence.md
