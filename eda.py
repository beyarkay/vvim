print("Imports...", end="")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime
pd.set_option('display.max_columns', None)
sns.set()
sns.set_style("darkgrid")
print("done")

# These are the characters actually covered by the sensors
# Finger notation is that as for piano. 5 is pinky, though to 1 for thumb
r1 = ["space"]
r2 = ["j", "m", "n", "b", "h", "y"]
r3 = [ "k", "y", "u", "i", "<", "(", "[" ]
r4 = ["l", ":", "[del]", "1", "o", "p", ">", ")", "]", "0", "_", "-", "+", "=", ",", "."]
r5 = [";", "[return]", "/", "?"]
COVERAGE = list(set(r2 + r3 + r4 + r5))
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
SENSOR_DESCRIPTIONS = {
    0: "kbd",
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
    13: "right-pinky-3",
}
def preprocess():
    print("Preprocessing data from sorted.log")
    # Read in data from tsv
    df = pd.read_csv("sorted.log", "\t", names=['micros', 'sensor', 'value'])    

    # Convert micros since epoch to datetime, 
    df['datetime'] = pd.to_datetime(df['micros'], unit='us')
    # Remove the irrelevant micros column
    del df['micros']
    # df['sensor'] = pd.to_numeric(df['sensor'])
    
    # Drop all the sensors we're not using
    df = df[df['sensor'].isin(SENSORS_IN_USE)]
    # Rename the sensors from indices to names
    df[['sensor']] = df[['sensor']].replace(to_replace=SENSOR_DESCRIPTIONS)
    print(df.head())
    # Describe the df so far
    print(df.describe(datetime_is_numeric=True))
    # Store the key presses separately from the sensor value readings
    sensors = df.loc[df['sensor'] != 'kbd']
    # The sensor's values are all numerical data readings
    sensors['value'] = pd.to_numeric(sensors['value'])
    ## Normalise the sensor values to have zero mean and unit variance
    #gb = sensors.groupby('sensor').describe()['value'][['mean', 'std']]
    #def zero_mean_unit_variance(row):
    #    if row['sensor'] == 'kbd':
    #        return row['value']
    #    else:
    #        s = pd.to_numeric(row['sensor'])
    #        return (pd.to_numeric(row['value']) -  gb.loc[s, 'mean']) / gb.loc[s, 'std']
    #df['value'] = df.apply(zero_mean_unit_variance, axis='columns')
    #sensors = df[df['sensor'] != 'kbd']
    #sensors['value'] = pd.to_numeric(sensors['value'])

    # Only keep keypresses that we've got sensor data for
    sensors_start = sensors['datetime'].min()
    sensors_end = sensors['datetime'].max()
    keys = df[(df['sensor'] == 'kbd') & (df['datetime'].between(sensors_start, sensors_end))]
    print("done")
    return df, sensors, keys

def main():
    df, sensors, keys = preprocess()
    text_y_val = sensors['value'].max()

    # Of those keys under coverage, find the most pressed one
    print(f"Coverage is: {COVERAGE}")
    covered = keys[keys['value'].isin(COVERAGE)]
    print("Keys ranked by number of key presses")
    print(covered['value'].value_counts())

    # TODO: Make the lines transparent, and auto-save
    keys_most_pressed = covered['value'].value_counts().index
    most_pressed = keys_most_pressed[0]
    for most_pressed in covered['value'].value_counts().index:
        print(f"Adjusting data to just show timedelta before and after the '{most_pressed}' key was pressed...", end="")
        times = covered.loc[covered['value'] == most_pressed, 'datetime']
        offset = pd.DataFrame()
        micros = 500_000
        for time in times:
            tmp = df.copy()
            tmp['datetime'] -= time
            tmp = tmp[np.abs(tmp['datetime']) < datetime.timedelta(microseconds=micros)]
            tmp['offset'] = time
            offset = offset.append(tmp)

        sensors_offset = offset[offset['sensor'] != 'kbd']
        sensors_offset['value'] = pd.to_numeric(sensors_offset['value'])
        keys_offset = offset[(offset['sensor'] == 'kbd')]
        plt.figure()
        num_sensors = len(sensors_offset['sensor'].unique())
        print(f"Sensors are: {sensors_offset['sensor'].unique()} ({num_sensors})")
        sensors_offset['datetime'] = pd.to_numeric(sensors_offset['datetime'])
        g = sns.FacetGrid(
            data=sensors_offset,
            col='sensor',
            col_order=SENSOR_ORDER,
            hue='offset',
            col_wrap=1,
            sharey=False,
            aspect=2
        )
        g.map(
            sns.lineplot,
            'datetime', 
            'value',
            estimator=None,
            alpha=0.5
        )
        g.set_titles(col_template="Sensor '{col_name}', " + f"{micros//1000}ms before and after '{most_pressed}' was pressed")
        g.set_xlabels("Time offset (100ms)")
        g.set_ylabels("Sensor Reading")
        # plt.text(x=0, y=sensors_offset['value'].max(), s=f'"{most_pressed}"', alpha=0.5)
        # plt.axvline(x=0, alpha=0.5, c='black', linewidth=1)

        # Finally, show the plot
        plt.savefig(f"plots/{most_pressed}_{micros//1000}ms_portrait.png", dpi=200)
        
        print("saved")
    print("done")




if __name__ == "__main__":
    main()
    print("Program Complete")

