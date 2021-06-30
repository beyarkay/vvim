print("Imports...", end="")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime

sns.set()
print("done")

# These are the characters actually covered by the sensors
# Finger notation is that as for piano. 5 is pinky, though to 1 for thumb
r1 = ["space"]
r2 = ["j", "m", "n", "b", "h", "y"]
r3 = [ "k", "y", "u", "i", "<", "(", "[" ]
r4 = ["l", ":", "[del]", "1", "o", "p", ">", ")", "]", "0", "_", "-", "+", "=", ",", "."]
r5 = [";", "[return]", "/", "?"]
COVERAGE = list(set(r2 + r3))
def preprocess():
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
    return df, sensors, keys

def main():
    df, sensors, keys = preprocess()
    text_y_val = sensors['value'].max()
    # print("Plotting sensor values over time...", end="")
    # # Plot the sensor values over time
    # sns.lineplot(x='datetime', y='value', hue='sensor', data=sensors)
    # # And include vertical lines to indicate each keypress
    # def plot_annotation(row, y):
    #     a = 0.7 if row['value'] in COVERAGE else 0.1
    #     plt.text(x=row['datetime'], y=y, s=row['value'], alpha=a)
    #     plt.axvline(x=row['datetime'], alpha=a, c='black', linewidth=1)
    # keys.apply(lambda row: plot_annotation(row, text_y_val), axis=1)

    # # Finally, show the plot
    # plt.show()
    # print("done")

    # Of those keys under coverage, find the most pressed one
    covered = keys[keys['value'].isin(COVERAGE)]
    print("Most pressed keys:")
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

        sensors_offset = offset[offset['sensor'] != 0]
        sensors_offset['value'] = pd.to_numeric(sensors_offset['value'])
        keys_offset = offset[(offset['sensor'] == 0)]
        plt.figure()
        sns.lineplot(x='datetime', y='value', hue='sensor', style='offset', data=sensors_offset, legend=False, alpha=0.5, palette=sns.color_palette("husl", 8)[:len(sensors_offset['sensor'].unique())])
        plt.text(x=0, y=sensors_offset['value'].max(), s=f'"{most_pressed}"', alpha=0.5)
        plt.axvline(x=0, alpha=0.5, c='black', linewidth=1)

        # Finally, show the plot
        plt.title(f"All Sensors {micros//1000}ms before and after '{most_pressed}' was pressed")
        plt.savefig(f"plots/{most_pressed}_{micros//1000}ms.png", dpi=200)
        
        print("saved")
    print("done")




if __name__ == "__main__":
    main()
    print("Program Complete")

