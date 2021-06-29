from sklearn.datasets import load_boston, load_digits
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import datetime

print("Preprocessing the data...", end="")
# Read in data from tsv
df = pd.read_csv("sorted.log", "\t", names=['micros', 'sensor', 'value'])    
# Convert micros since epoch to datetime, 
df['datetime'] = pd.to_datetime(df['micros'], unit='us')
# Remove the irrelevant micros column
del df['micros']
# the sensor column should be numeric (it's an int from 0 to n)
df['sensor'] = pd.to_numeric(df['sensor'])
# Describe the df so far
print(df.describe(datetime_is_numeric=True))
# Store the key presses separately from the sensor value readings
sensors = df[df['sensor'] != 0]
# The sensor's values are all numerical data readings
sensors['value'] = pd.to_numeric(sensors['value'])
# Only keep keypresses that we've got sensor data for
sensors_start = sensors['datetime'].min()
sensors_end = sensors['datetime'].max()
keys = df[(df['sensor'] == 0) & (df['datetime'].between(sensors_start, sensors_end))]
print("done")

digits = load_digits()
X_digits, Y_digits = digits.data, digits.target

NUM_SAMPLES = 4*16          # Multiply by 4 because there are 4 sensors
DURATION_US = 500_000

sensors.sort_values(['datetime', 'sensor'], inplace=True)
keys.sort_values('datetime', inplace=True)
# Remove all keys that don't have at least NUM_SAMPLES values before them
sensors_start = sensors['datetime'].min() + datetime.timedelta(microseconds=DURATION_US)
sensors_end = sensors['datetime'].max()
COVERAGE = list('bhijkmnuy')
keys = keys[(keys['datetime'].between(sensors_start, sensors_end)) & (keys['value'].isin(COVERAGE))]

def extract_sensor_data(row):
    sensors_start = row['datetime'] - datetime.timedelta(microseconds=DURATION_US)
    sensor_measurements = list(sensors.loc[sensors['datetime'].between(sensors_start, row['datetime']), 'value'])
    idx = np.round(np.linspace(0, len(sensor_measurements) - 1, NUM_SAMPLES)).astype(int)
    sensor_measurements = np.array(sensor_measurements)[idx]
    return [row['value'], row['datetime']] + list(sensor_measurements)


# Collect all the samples into one dataframe
all_data = keys.apply(extract_sensor_data, axis=1, result_type='expand')
all_data.rename(columns={0:'value', 1:'datetime'}, inplace=True)

# TODO might need to map these to numbers
y_data = np.array(all_data['value'])
X_data = np.array(all_data.loc[:, range(2, NUM_SAMPLES+2)])

print('Dataset Size : ',X_digits.shape, Y_digits.shape)
print('Dataset Size : ',X_data.shape, y_data.shape)

#X_train, X_test, Y_train, Y_test = train_test_split(
#    X_digits,
#    Y_digits,
#    train_size=0.80,
#    test_size=0.20,
#    random_state=123
#)
X_train, X_test, Y_train, Y_test = train_test_split( X_data, y_data, train_size=0.80, test_size=0.20, random_state=123)
print('Train/Test Sizes : ', X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)



n_samples = X_data.shape[0]
n_features = X_data.shape[1]

params = {
    'n_estimators': [400, 800,],
    'max_depth': [2, 5, ],
    'min_samples_split': [0.4, 0.5],
    'min_samples_leaf': [0.5, 1, 16],
    'criterion': ['friedman_mse',],
    'max_features': ['sqrt', 'log2', 0.5,],
}
# Best Accuracy Through Grid Search : 0.776
# Best Parameters :  {'criterion': 'friedman_mse', 'max_depth': 5, 'min_samples_split': 0.5, 'n_estimators': 200}

# Best Accuracy Through Grid Search : 0.785
# Best Parameters :  {'criterion': 'friedman_mse', 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 0.5, 'n_estimators': 200}

# Best Accuracy Through Grid Search : 0.787
# Best Parameters :  {'criterion': 'friedman_mse', 'max_depth': 5, 'max_features': 0.5, 'min_samples_leaf': 1, 'min_samples_split': 0.5, 'n_estimators': 400}

# Best Accuracy Through Grid Search : 0.793
# Best Parameters :  {'criterion': 'friedman_mse', 'max_depth': 2, 'max_features': 0.5, 'min_samples_leaf': 16, 'min_samples_split': 0.5, 'n_estimators': 400}



grad_boost_classif_grid = GridSearchCV( GradientBoostingClassifier(random_state=1), param_grid=params, n_jobs=-1, cv=3, verbose=5)
grad_boost_classif_grid.fit(X_train,Y_train)

print('Train Accuracy : %.3f'%grad_boost_classif_grid.best_estimator_.score(X_train, Y_train))
print('Test Accuracy : %.3f'%grad_boost_classif_grid.best_estimator_.score(X_test, Y_test))
print('Best Accuracy Through Grid Search : %.3f'%grad_boost_classif_grid.best_score_)
print('Best Parameters : ',grad_boost_classif_grid.best_params_)

