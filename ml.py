print("Imports...", end="")
from sklearn.datasets import load_boston, load_digits
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import os
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
import numpy as np 
import datetime
import matplotlib.pyplot as plt
print("done")

# =========
# Constants
# =========

# These are the characters actually covered by the sensors
# Finger notation is that as for piano. 5 is pinky, though to 1 for thumb
r1 = [" "]
r2 = ["j", "m", "n", "b", "h", "y"]
r3 = [ "k", "y", "u", "i", "<", "(", "[" ]
r4 = ["l", ":", "[del]", "1", "o", "p", ">", ")", "]", "0", "_", "-", "+", "=", ",", "."]
r5 = [";", "[return]", "/", "?"]
misc = ["NoKey"] #, "[left-ctrl]", "[esc]", "a", "e"]
COVERAGE = list(set(r1 + r2 + r3 + r4 + r5 + misc))
NUM_SAMPLES = 8*64         # Multiply by 8 because there are 8 sensors
DURATION_US = 500_000

if os.path.exists("df.pkl"):
    print("Reading data from `df.pkl`")
    df = pd.read_pickle("df.pkl")
else:
    # ================
    # Read in the data
    # ================
    print("Preprocessing the data from sorted.log...", end="")
    # Read in data from tsv
    df = pd.read_csv("sorted.log", "\t", names=['micros', 'sensor', 'value'])    
    # Ensure the sensor readings are in the correct range
    df = df.loc[df['sensor'].between(0, 13), :]
    # Convert micros since epoch to datetime, 
    df['datetime'] = pd.to_datetime(df['micros'], unit='us')
    # Remove the irrelevant micros column
    del df['micros']
    # the sensor column should be numeric (it's an int from 0 to n)
    df['sensor'] = pd.to_numeric(df['sensor'])
    # Describe the df so far
    print(df.describe(datetime_is_numeric=True))
    # TODO there's a 14th sensor that gets included for some reason...
    print("done")

    # ----------------------------------------
    # Impute data where there's no key pressed
    # ----------------------------------------

    print("Imputing data when there's no key pressed...")
    # Only test with a small portion of the data to speed up the process
    # df = df[df['datetime'] < pd.Timestamp("2021-07-12 19:25:00.000")]
    # Group all the data by the sensor readings were collected
    gb = df.groupby('datetime')
    # create a mask of every datetime that a key wasn't pressed but that we do have
    # sensor readings for
    needs_none_key = gb['sensor'].unique().apply(lambda row: 0 not in row)
    # Use that mask to get a series of every datetime a key wasn't pressed.
    nnk_datetimes = pd.Series(needs_none_key[needs_none_key].index)
    def add_NoKey_if_no_key(groupby):
        if groupby['datetime'].isin(nnk_datetimes).all():
            dummy_series = pd.Series({'sensor':0, 'value': 'NoKey'});
            return groupby.append(dummy_series, ignore_index=True);
        else:
            return groupby
    # Add a 'NoKey' dummy keypress to every datetime without a real keypress
    print("Applying imputing function...")
    df = gb.apply(add_NoKey_if_no_key).drop('datetime', axis='columns').reset_index().drop('level_1', axis='columns')
    print("Done imputing")
    df.to_pickle("df.pkl")

# ------------------------------------------------------
# Create `keys` and `sensors` dataframes for convenience
# ------------------------------------------------------

# Store the key presses separately from the sensor value readings
sensors = df[df['sensor'] != 0]
# The sensor's values are all numerical data readings
sensors['value'] = pd.to_numeric(sensors['value'])
# Only keep keypresses that we've got sensor data for
sensors_start = sensors['datetime'].min() + datetime.timedelta(microseconds=DURATION_US)
sensors_end = sensors['datetime'].max()
print(f"df:\n{df.loc[df['sensor'] == 0, 'value'].value_counts()}")
keys = df[
    (df['sensor'] == 0) & 
    (df['datetime'].between(sensors_start, sensors_end)) & 
    (df['value'].isin(COVERAGE))
]
perc_NoKey = list(keys['value'].value_counts())[1] / list(keys['value'].value_counts())[0]
print(perc_NoKey)
keys = keys[
        ((keys['value'] == 'NoKey') & (np.random.random(len(keys)) < perc_NoKey)) | 
        (keys['value'] != 'NoKey')
        ]
print(f"Keys:\n{keys['value'].value_counts()}")
sensors.sort_values(['datetime', 'sensor'], inplace=True)
keys.sort_values('datetime', inplace=True)


print(f"Num sensor readings: \n{sensors['sensor'].value_counts()}")

def extract_sensor_data(row):
    """ Converts rows to be ready for a boosted forest model """
    sensors_start = row['datetime'] - datetime.timedelta(microseconds=DURATION_US)
    sensor_measurements = list(sensors.loc[sensors['datetime'].between(sensors_start, row['datetime']), 'value'])
    idx = np.round(np.linspace(0, len(sensor_measurements) - 1, NUM_SAMPLES)).astype(int)
    sensor_measurements = np.array(sensor_measurements)[idx]
    return [row['value'], row['datetime']] + list(sensor_measurements)


print(f"Keys:\n{keys['value'].value_counts()}")

if os.path.exists("all_data.pkl"):
    print("Reading data from `all_data.pkl`")
    all_data = pd.read_pickle("all_data.pkl")
else:
    # Collect all the samples into one dataframe
    all_data = keys.apply(extract_sensor_data, axis=1, result_type='expand')
    all_data.rename(columns={0:'value', 1:'datetime'}, inplace=True)
    all_data.to_pickle("all_data.pkl")

y_data = np.array(all_data['value'])
X_data = np.array(all_data.loc[:, range(2, NUM_SAMPLES+2)])

print('Dataset Size : ',X_data.shape, y_data.shape)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.80, test_size=0.20, random_state=123)
print('Train/Test Sizes : ', X_train.shape, X_test.shape, y_train.shape, y_test.shape)

n_samples = X_data.shape[0]
n_features = X_data.shape[1]

should_train_model = False
should_gridsearch = True

if should_gridsearch:
    params = {
        'n_estimators': [100, 200, 400,],
        'max_depth': [1, 2, 5, ],
        'min_samples_split': [0.4, 0.5],
        'min_samples_leaf': [0.5, 1, 16],
        'criterion': ['friedman_mse', 'mse'],
        'max_features': ['sqrt', 'log2', 0.5,],
    }
    grad_boost_classif_grid = GridSearchCV( GradientBoostingClassifier(random_state=1), param_grid=params, n_jobs=-1, cv=3, verbose=5)
    grad_boost_classif_grid.fit(X_train,y_train)
    print('Train Accuracy : %.3f'%grad_boost_classif_grid.best_estimator_.score(X_train, y_train))
    print('Test Accuracy : %.3f'%grad_boost_classif_grid.best_estimator_.score(X_test, y_test))
    print('Best Accuracy Through Grid Search : %.3f'%grad_boost_classif_grid.best_score_)
    print('Best Parameters : ',grad_boost_classif_grid.best_params_)
    from sklearn.metrics import plot_confusion_matrix
    np.set_printoptions(precision=1)
    disp = plot_confusion_matrix(best, X_test, y_test,
                                    cmap=plt.cm.Blues,
                                    normalize='true')
    disp.ax_.set_title("Normalized Confusion Matrix")

    print("Normalized Confusion Matrix")
    print(disp.confusion_matrix)
    
    # plt.savefig("images/confusion.png")
    plt.show()

elif should_train_model:
    print("Training Model")
    best = GradientBoostingClassifier(random_state=1, 
            criterion='friedman_mse', max_depth=5, max_features='sqrt', min_samples_leaf=1, min_samples_split=0.4, n_estimators=100, verbose=1
    )
    best.fit(X_train,y_train)
    print('Train Accuracy : %.3f'%best.score(X_train, y_train))
    print('Test Accuracy : %.3f'%best.score(X_test, y_test))

    from sklearn.metrics import plot_confusion_matrix
    np.set_printoptions(precision=1)
    disp = plot_confusion_matrix(best, X_test, y_test,
                                    cmap=plt.cm.Blues,
                                    normalize='true')
    disp.ax_.set_title("Normalized Confusion Matrix")

    print("Normalized Confusion Matrix")
    print(disp.confusion_matrix)
    
    # plt.savefig("images/confusion.png")
    plt.show()


    import pickle
    with open('model.pkl','wb') as f:
            pickle.dump(best,f)
else:
    import pickle

    with open('model.pkl','rb') as f:
        best = pickle.load(f)
    print('Train Accuracy : %.3f'%best.score(X_train, y_train))
    print('Test Accuracy : %.3f'%best.score(X_test, y_test))

    from sklearn.metrics import plot_confusion_matrix
    np.set_printoptions(precision=1)

    plt.rcParams.update({'font.size': 6})
    disp = plot_confusion_matrix(best, X_test, y_test,
            values_format='.2f', 
            cmap=plt.cm.Blues,
            normalize='true')
    disp.ax_.set_title("Normalized Confusion Matrix")

    print("Normalized Confusion Matrix")
    print(disp.confusion_matrix)
    
    plt.savefig("images/confusion.png")
    plt.show()

    with open('model.pkl','wb') as f:
            pickle.dump(best,f)


