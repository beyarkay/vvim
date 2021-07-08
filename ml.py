from sklearn.datasets import load_boston, load_digits
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# =========
# Constants
# =========

# These are the characters actually covered by the sensors
# Finger notation is that as for piano. 5 is pinky, though to 1 for thumb
r1 = ["space"]
r2 = ["j", "m", "n", "b", "h", "y"]
r3 = [ "k", "y", "u", "i", "<", "(", "[" ]
r4 = ["l", ":", "[del]", "1", "o", "p", ">", ")", "]", "0", "_", "-", "+", "=", ",", "."]
r5 = [";", "[return]", "/", "?"]
COVERAGE = list(set(r2 + r3))
RHS = list(set(r1 + r2 + r3 + r4 + r5))
should_use_rhs = True
# ======================
# Preprocessing the data
# ======================

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

# ======================
# Load in the glove data
# ======================

NUM_SAMPLES = 4*128         # Multiply by 4 because there are 4 sensors
DURATION_US = 500_000

sensors.sort_values(['datetime', 'sensor'], inplace=True)
keys.sort_values('datetime', inplace=True)
# Remove all keys that don't have at least NUM_SAMPLES values before them
sensors_start = sensors['datetime'].min() + datetime.timedelta(microseconds=DURATION_US)
sensors_end = sensors['datetime'].max()
if not should_use_rhs:
    keys = keys[(keys['datetime'].between(sensors_start, sensors_end)) & (keys['value'].isin(COVERAGE))]
else:
    keys = keys[(keys['datetime'].between(sensors_start, sensors_end)) & (keys['value'].isin(RHS))]

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

print('Dataset Size : ',X_data.shape, y_data.shape)

X_train, X_test, y_train, y_test = train_test_split( X_data, y_data, train_size=0.80, test_size=0.20, random_state=123)
print('Train/Test Sizes : ', X_train.shape, X_test.shape, y_train.shape, y_test.shape)



n_samples = X_data.shape[0]
n_features = X_data.shape[1]


should_train_model = False
should_gridsearch = False

if should_gridsearch:
    params = {
        'n_estimators': [100, 200, 400, 800,],
        'max_depth': [1, 2, 5, ],
        'min_samples_split': [0.4, 0.5],
        'min_samples_leaf': [0.5, 1, 16],
        'criterion': ['friedman_mse', 'mse'],
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
    grad_boost_classif_grid.fit(X_train,y_train)
    print('Train Accuracy : %.3f'%grad_boost_classif_grid.best_estimator_.score(X_train, y_train))
    print('Test Accuracy : %.3f'%grad_boost_classif_grid.best_estimator_.score(X_test, y_test))
    print('Best Accuracy Through Grid Search : %.3f'%grad_boost_classif_grid.best_score_)
    print('Best Parameters : ',grad_boost_classif_grid.best_params_)
elif should_train_model:
    print("training model")

    best = GradientBoostingClassifier(random_state=1, 
            criterion='friedman_mse', max_depth=5, max_features='sqrt', min_samples_leaf=1, min_samples_split=0.4, n_estimators=800, verbose=1
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


