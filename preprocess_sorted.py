print("Imports...", end="")
import pandas as pd
import datetime
pd.set_option('display.max_columns', None)
print("done")

# These are the characters actually covered by the sensors
# Finger notation is that as for piano. 5 is pinky, though to 1 for thumb
r1 = [' ']
r2 = ["j", "m", "n", "b", "h", "y"]
r3 = [ "k", "y", "u", "i", "<", "(", "[", "*", "8", "(", "9"]
r4 = ["l", ":", "[del]", "o", "p", ">", ")", "]", "_", ")", "0", "-", "+", "=",
        ",", ".", "6", "^", "7", "&"]
r5 = [";", ":", "'", '"', "[return]", "/", "?", "\\", "|", "{", "}",
        "[right]", "[left]", "[up]", "[right]"]
# uppercase = [c.upper() for c in r2 + r3 + r4 + r5]
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
# Store the key presses separately from the sensor value readings
sensors = df.loc[df['sensor'] != 'kbd']
# The sensor's values are all numerical data readings
sensors.loc[:, 'value'] = pd.to_numeric(sensors.loc[:, 'value'])
# Only keep keypresses that we've got sensor data for
sensors_start = sensors['datetime'].min()
sensors_end = sensors['datetime'].max()
keys = df[(df['sensor'] == 'kbd') & (df['datetime'].between(sensors_start, sensors_end))]
# Lower-case all the values because we don't know if a shift key has been pressed
keys.value = keys.value.apply(lambda x: x.lower())

# Remove outliers
l, h = sensors.value.quantile([0.0001, 0.9999])
sensors = sensors[sensors.value.between(l, h)]
# sns.lineplot(data=sensors, x='datetime', y='value', hue='sensor') # Plot the full timeline

# resample
period_size =pd.Timedelta(50, unit="ms") 
sensors = sensors.groupby('sensor').resample(period_size, on='datetime').median().reset_index()
keys = keys.set_index('datetime').resample(period_size).first().reset_index()

value_counts = keys.value_counts('value')
keys = keys[keys.value.isin(COVERAGE)]

now_as_iso = datetime.datetime.isoformat(datetime.datetime.now())[:-16]
keys.to_pickle(f"keys_{now_as_iso}.pkl")
print("WARNING: Keys typed but not under coverage:")
print(sorted(list(value_counts[~value_counts.index.isin(COVERAGE)].index)))
print("Keys Data:")
print("\tFrequency of key presses (where for the RH)\n")
print(value_counts[value_counts.index.isin(COVERAGE)])
print("\n\tDistribution of datetimes\n")
print(keys['datetime'].describe(datetime_is_numeric=True))
sensors.to_pickle(f"sensors_{now_as_iso}.pkl")
print("\n\nSensors Data:")
print("\tDescribe for each sensor:")
print(sensors.groupby('sensor')['value'].describe()[['count', 'mean', 'std', 'min', '50%', 'max']])

itimes = keys.loc[keys.value == 'i', 'datetime']
isensors = sensors[sensors.datetime.isin(itimes)]
print("Making chart...")
sns.lineplot(data=sensors, x='datetime', y='value', hue='sensor', lw=0.5)
plt.show()

Xy = sensors
# TODO: this is super slow ):
Xy['key'] = Xy.datetime.apply(lambda x: (keys.loc[keys.datetime==x, 'value'].unique()[0] if (keys.datetime==x).any() else np.NaN))
freq_keys = Xy['key'].value_counts().index.to_list()[:10]

# Faceted boxenplot
g = sns.FacetGrid(Xy[Xy.key.isin(freq_keys)], col='sensor', col_wrap=2)
g.map(sns.boxplot, 'value', 'key', palette='Set3')
plt.show()
print(f"Saved to keys_{now_as_iso}.pkl and sensors_{now_as_iso}.pkl")

