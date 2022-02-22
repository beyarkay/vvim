print("Imports...", end="")
import seaborn as sns
import datetime
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np 
import os
import pandas as pd
sns.set()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
print("done")

# Load in some parameters constant across the various scripts
with open('params.json', 'r') as f:
    params = json.load(f)

typeable = []
for v in params.get('fingers-to-keys').values():
    typeable.extend(v)

typeable = list(set(typeable))

# Load in the keys and sensors datasets
keys = pd.read_pickle('keys_2021-09-29.pkl').sort_values('datetime')
keys = keys[keys.value.isin(typeable)]
sensors = pd.read_pickle('sensors_2021-09-29.pkl').sort_values('datetime')
# Take only the subset of sensor readings occuring in the instant a key
# was pressed
print("using kbd instead of keyboard")
sub_keys = keys.loc[keys.sensor=='kbd', ['datetime', 'value']]
sub_sens = sensors[sensors.datetime.isin(sub_keys.datetime)]
X_df = sub_sens.pivot(index='datetime', columns='sensor', values='value') 
y_df = sub_keys.set_index('datetime')
# Replace " " with "[space]" for clarity
y_df.loc[y_df.value==' ', 'value'] = "[space]"


# ----------------------------------------------------
# Plot a polar plot for every frequently occurring key
# TODO: groupby each key and then normalise, so that
# you can see when a measurement is out of normal
# TODO: Instead of plotting every point, just plot 
# the mean
# ----------------------------------------------------
df = pd.concat([y_df, X_df], axis=1)
df.rename(columns={'value':'key'}, inplace=True)
# Convert df from wide to long format
values = df.columns.to_list()
values.remove('key')
# Rename the value column because melt doesn't work well with it.
df = df.melt(id_vars=['key'], value_vars=values, ignore_index=False).reset_index()
common_keys = df.value_counts('key').head(20).index.to_list()
df = df[df.key.isin(common_keys)]
indexes = [int(s) for s in params.get('sensors-in-use') if s != '0']
ticklabels = pd.Series(params.get('index-to-name').values())[indexes]
# reverse the labels to go from index to pinky, left to right
ticklabels = ticklabels[::-1]
mpl.rcParams['xtick.labelsize'] = 8
short_ticklables = [tl.replace("right", "r")
                      .replace("middle", "m")
                      .replace("ring", "r")
                      .replace("pinky", "p")
                      .replace("index", "i") 
                      .replace("-", "") for tl in ticklabels]
# Convert from variable to theta to aid with plotting
num_variables = len(df.variable.unique())
per_segment = 1 * np.pi / num_variables
df['theta'] = df['variable'].apply(lambda x: np.where(ticklabels == x)[0][0] * per_segment + per_segment * 0.5)
# Visualisation tweaks
g = sns.FacetGrid(df, col="key", hue="key", col_wrap=5,
                  subplot_kws=dict(projection='polar'),
                  sharex=False, sharey=False, despine=False)
g.map(sns.scatterplot, "theta", "value")
for ax in g.axes:
    ax.set_thetamin(0)
    ax.set_thetamax(180)

g.set(
    xticks=[i*per_segment + 0.5*per_segment for i in range(num_variables)],
    xticklabels=short_ticklables, 
    yticklabels=[])
g.set_titles(col_template="{col_name}")
g.set_axis_labels("", "")
plt.tight_layout()
plt.show()


