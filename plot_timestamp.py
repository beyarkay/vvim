print("Imports...", end="")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime
import sys
pd.set_option('display.max_columns', None)
sns.set()
sns.set_style("darkgrid")
print("done")

# These are the characters actually covered by the sensors
# Finger notation is that as for piano. 5 is pinky, though to 1 for thumb
r1 = ["space"];
r2 = ["j", "m", "n", "b", "h", "y"];
r3 = [ "k", "y", "u", "i", "<", "(", "[" ];
r4 = ["l", ":", "[del]", "1", "o", "p", ">", ")", "]", "0", "_", "-", "+", "=", ",", "."];
r5 = [";", "[return]", "/", "?"];
COVERAGE = list(set(r2 + r3 + r4 + r5));
# Note that the arduino is zero indexed, but the python script has to be
# one-indexed as the zero-th 'sensor' is the string indicating which key was
# pressed
SENSORS_IN_USE = ["0", "2", "3", "5", "6", "8", "9", "11", "12"]
SENSOR_ORDER = ["right-index-1",
    "right-index-2",
    "right-middle-1",
    "right-middle-2",
    "right-ring-1",
    "right-ring-2",
    "right-pinky-1",
    "right-pinky-2",
]
SENSOR_DESCRIPTIONS = { 0: "kbd",
    1: "right-thumb-3",
    2: "right-index-1",
    3: "right-index-2",
    4: "right-index-3",
    5: "right-middle-1",
    6: "right-middle-2",
    7: "right-middle-3",
    8: "right-ring-1",
    9: "right-ring-2",
    10: "right-ring-3",
    11: "right-pinky-1",
    12: "right-pinky-2",
    13: "right-pinky-3"}
def preprocess():
    print("Preprocessing data from sorted.log")
    # Read in data from tsv
    df = pd.read_csv("sorted.log", "\t", names=['micros', 'sensor', 'value'])    

    # Convert micros since epoch to datetime, 
    df['datetime'] = pd.to_datetime(df['micros'], unit='us')
    # Remove the irrelevant micros column
    del df['micros']
    # Drop all the sensors we're not using
    df = df[df['sensor'].isin(SENSORS_IN_USE)]
    # Rename the sensors from indices to names
    df[['sensor']] = df[['sensor']].replace(to_replace=SENSOR_DESCRIPTIONS)
    # Describe the df so far
    print(df.describe(datetime_is_numeric=True))
    # Store the key presses separately from the sensor value readings
    sensors = df[df['sensor'] != 'kbd']
    # The sensor's values are all numerical data readings
    sensors['value'] = pd.to_numeric(sensors['value'])
    # Only keep keypresses that we've got sensor data for
    sensors_start = sensors['datetime'].min()
    sensors_end = sensors['datetime'].max()
    keys = df[(df['sensor'] == 'kbd') & (df['datetime'].between(sensors_start, sensors_end))]
    print("done")
    return df, sensors, keys

def main():
    df, sensors, keys = preprocess()
    def plt_keys(row): plt.text(x=row['datetime'], y=sensors_in_range['value'].max(), s=f"'{row['value']}'", alpha=0.5); plt.axvline(x=row['datetime'], alpha=0.5, c='black', linewidth=1)

    if len(sys.argv) == 1:
        print("A timestamp needs to be provided. For example:")
        print(keys['datetime'].head())
        print(keys['datetime'].tail())
        sys.exit(1)
    timestamp = pd.Timestamp(sys.argv[1])
    ms_either_side = int(sys.argv[2]) if len(sys.argv) == 3 else 500 
    extent = [timestamp - pd.Timedelta(ms_either_side, unit='ms'), timestamp + pd.Timedelta(ms_either_side, unit='ms')]
    sensors_in_range = sensors[sensors['datetime'].between(*extent)]
    keys_in_range = keys[keys['datetime'].between(*extent)]
    print(sensors_in_range)
    sns.lineplot(x='datetime', y='value', hue='sensor', style='sensor', data=sensors_in_range, markers=True)
    keys_in_range.apply(plt_keys, axis='columns')

    plt.show()
    print("done")


if __name__ == "__main__":
    main()
    print("Program Complete")

