print("Imports...", end="")
import datetime
import csv
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
sns.set()
pd.set_option('display.max_columns', None)
print("done")

with open('params.json', 'r') as f:
    params = json.load(f)

typeable = []
for _, v in params.get('fingers-to-keys').items():
    typeable.extend(v)

typeable = list(set(typeable))

print("Preprocessing data from sorted.log")
# -------------------------------------------
# Read in and clean up the original dataframe
# -------------------------------------------
# Read in data from tsv. Use python engine because it's more feature complete
df = pd.read_csv("sorted.log", "\t", names=['micros', 'sensor', 'value'], quoting=csv.QUOTE_NONE)
print("Dataframe read in successfully")
# Convert micros since epoch to datetime,
df['datetime'] = pd.to_datetime(df['micros'], unit='us')
# Remove the irrelevant micros column
del df['micros']
# Drop all the sensors we're not using
df = df[df['sensor'].isin(params.get('sensors-in-use'))]
# Rename the sensors from indices to names
replacers = {int(k): v for k, v in params.get('index-to-name').items()}
df[['sensor']] = df[['sensor']].replace(to_replace=replacers)

# -------------------------------
# Calculate the Sensors Dataframe
# -------------------------------
print("Calculating the Sensors Dataframe")
# Store the key presses separately from the sensor value readings
sensors = df.loc[df['sensor'] != 'keyboard']
# The sensor's values are all numerical data readings
sensors.loc[:, 'value'] = pd.to_numeric(sensors.loc[:, 'value'])
# Remove outliers
l, h = sensors.value.quantile([0.0001, 0.9999])
sensors = sensors[sensors.value.between(l, h)]

# resample
period_size = 10 if len(sys.argv) != 2 else int(sys.argv[1])
period_sizetd = pd.Timedelta(period_size, unit="ms")
# The data might be spread out over days or weeks which makes resampling for
# every 10ms very RAM intensive. Rather split up the data by date and resample
# on the parts
sensors['date'] = sensors.datetime.dt.date
gb = sensors.groupby('date')
sensor_dfs = [gb.get_group(group) for group in gb.groups]
print("Resampling the sensors dataframe")
resampled_sensors = []
for sensor_df in sensor_dfs:
    resampled_sensors.append(sensor_df.groupby('sensor').resample(period_sizetd, on='datetime').median().reset_index())

sensors = pd.concat(resampled_sensors)

# Remove any time points which have no sensor readings.
sensors = sensors.dropna()
now_as_iso = datetime.datetime.isoformat(datetime.datetime.now())[:-16]
sensors_path = f"sensors-{now_as_iso}-{period_size}ms.pkl"
sensors.to_pickle(f"sensors-{now_as_iso}-{period_size}ms.pkl")
print(f"Saved sensors as {sensors_path}")
del sensors

# ----------------------------
# Calculate the keys Dataframe
# ----------------------------
print("Calculating the keys dataframe")
# Get a df with the data
keys = df.loc[(df['sensor'] == 'keyboard') & (df['value'].isin(typeable))]

# Lower-case all the values because we don't know if a shift key has been pressed
def fancy_lower(x):
    """ lowercase _everything_, as though the shift key doesn't exist"""
    if x.isalpha():
        return x.lower()
    if x in params.get('lower-dict').keys():
        return params.get('lower-dict').get(x)
    return x

print("Lower casing all the key presses")
keys.value = keys.value.apply(fancy_lower)
# The data might be spread out over days or weeks which makes resampling for
# every 10ms very RAM intensive. Rather split up the data by date and resample
# on the parts
keys['date'] = keys.datetime.dt.date
gb = keys.groupby('date')
key_dfs = [gb.get_group(group) for group in gb.groups]
print("Re sampling the keys dataframe")
resampled_keys = []
for key_df in key_dfs:
    resampled_keys.append(key_df.set_index('datetime').resample(period_sizetd).first().reset_index().dropna())
keys = pd.concat(resampled_keys)
del keys['date']
print("Saving the dataframes")
keys_path = f"keys-{now_as_iso}-{period_size}ms.pkl"
keys.to_pickle(keys_path)
print(f"Saved keys as {keys_path}")

# print(f"Total Readings from keyboard: "
#         + str(df[(df['sensor'] == 'keyboard')].value.notna().sum()))
# print(f"Total Readings from sensors: ")
# print(df[df['sensor'] != 'keyboard'].groupby('sensor').count().value)


# print("Making chart...")
# if input("Make chart?"):
#     sns.lineplot(data=sensors, x='datetime', y='value', hue='sensor', lw=0.5)
#     def plt_keys(row):
#         plt.text(
#                 x=row['datetime'],
#                 y=sensors['value'].max(),
#                 s=f"'{row['value']}'",
#                 alpha=0.5
#         )
#         plt.axvline(x=row['datetime'], alpha=0.5, c='black', linewidth=1)
#
#     keys.apply(plt_keys, axis='columns')
#     plt.show()

# Xy = sensors
# # TODO: this is super slow ):
# print("Starting slow transformation of the data's datetimes")
# Xy['key'] = Xy.datetime.apply(lambda x: (keys.loc[keys.datetime==x, 'value'].unique()[0] if (keys.datetime==x).any() else np.NaN))
# freq_keys = Xy['key'].value_counts().index.to_list()[:10]

# Faceted boxenplot
# g = sns.FacetGrid(Xy[Xy.key.isin(freq_keys)], col='sensor', col_wrap=2)
# g.map(sns.boxplot, 'value', 'key', palette='Set3')
# plt.show()

