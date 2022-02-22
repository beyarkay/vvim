print("Imports...", end="")
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
sns.set()
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
print("done")
params = {}
typeable = []

# TODO Check residuals to see what the model is lacking
# TODO Try adding in more time steps, as opposed to just the very instant of
# pressing the key

def init():
    with open('params.json', 'r') as f:
        params = json.load(f)

    def fancy_lower(x):
        """ lowercase _everything_, as though the shift key doesn't exist"""
        if x.isalpha():
            return x.lower()
        if x in params.get('lower-dict').keys():
            return params.get('lower-dict').get(x)
        return x

    for _, v in params.get('fingers-to-keys').items():
        typeable.extend([fancy_lower(value) for value in v])

    typeable = [item for item in list(set(typeable)) if item not in params.get('untypeable')]


# Load in some parameters constant across the various scripts
def load_dataset(
        period_ms=10,
        num_periods=10,
        min_replicates=50,
        dataset_suffix='2021-09-29-10ms'):
    # Load in the keys and sensors datasets
    keys = pd.read_pickle(f'data/keys-{dataset_suffix}.pkl').sort_values('datetime')
    keys = keys[keys.value.isin(typeable)]
    sensors = pd.read_pickle(f'data/sensors-{dataset_suffix}.pkl').sort_values('datetime')
    # Take only the subset of sensor readings occurring in the instant a key
    # was pressed
    sub_keys = keys.loc[keys.sensor=='keyboard', ['datetime', 'value']]
    period = pd.Timedelta(period_ms, unit='ms')
    dfs = []
    for i in range(0, num_periods):
        datetimes = pd.Series(dtype='datetime64[ns]')
        delta = period * i
        datetimes = sub_keys.datetime - delta
        sub_sens = sensors.loc[sensors.datetime.isin(datetimes), :]
        sub_sens['neg_offset'] = delta
        sub_sens['datetime'] += delta
        dfs.append(sub_sens)
    sub_sens = pd.concat(dfs)
    assert len(sub_sens[sub_sens.isna().any(axis=1)]) == 0
    X_df = sub_sens.pivot(index='datetime', columns=['sensor', 'neg_offset'], values='value')
    y_df = sub_keys.set_index('datetime')
    # Replace " " with "[space]" for clarity
    y_df.loc[y_df.value==' ', 'value'] = "[space]"
    # Remove observations with less than 10 replicates
    vc = y_df.value_counts()
    has_enough_replicates = (vc[vc >= min_replicates]).reset_index().value.to_list()
    y_df = y_df[y_df.value.isin(has_enough_replicates)]
    # Make sure the X's match the y's
    X_df.dropna(inplace=True)
    X_df = X_df[X_df.index.isin(y_df.index)]
    y_df = y_df[y_df.index.isin(X_df.index)]
    # Check that the dataframes are of the correct dimensions for each other
    assert (X_df.index == y_df.index).all()
    # Convert dataframes to ndarrays for sklearn
    X = X_df.to_numpy()
    y = y_df.to_numpy().ravel()
    # Check that there aren't any NaN values
    assert not np.isnan(X).any()
    # Build training and testing data
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

# =============================
# Attempt to train a MLP model
# =============================
# Imports
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.utils.fixes import loguniform

num_periods_list = [2, 10, 20, 40, 80]
period_ms_list = [10, 20, 40, 80]
hidden_layer_sizes = [(500), (250), (125), (75), (25, 25), (50, 50), (25, 25, 25)]
history_max_ms = 1000
configs = []
best_test_acc = 0.0
best_model = None
for num_periods in num_periods_list:
    print(f"num_periods={num_periods}")
    for period_ms in period_ms_list:
        total_history_ms = (num_periods - 1) * period_ms
        # if total_history_ms >= history_max_ms:
        #     print(f"\tSkipping: total_history_ms={total_history_ms}")
        #     continue
        print(f"\tperiod_ms={period_ms}")
        X_train, X_test, y_train, y_test = load_dataset(
                period_ms=period_ms, num_periods=num_periods,
                dataset_suffix='2021-10-02-10ms'
        )
        for hls in hidden_layer_sizes:
            print(f"\t\thls={hls}", end="", flush=True)
            config={'period_ms': period_ms,
                    'num_periods': num_periods,
                    'total_history_ms': total_history_ms,
                    'hls': hls}
            # First scale the data
            scaler = StandardScaler()
            _ = scaler.fit(X_train) # Don't cheat - fit only on training data
            X_trn_scaled = scaler.transform(X_train)
            # apply same transformation to test data
            X_tst_scaled = scaler.transform(X_test)
            mlp_clf = MLPClassifier(activation='tanh', hidden_layer_sizes=hls, alpha=0.001, max_iter=5000, random_state=42, solver='adam', tol=0.00001) #, verbose=2)
            _ = mlp_clf.fit(X_trn_scaled, y_train)
            train_acc = metrics.accuracy_score(y_train, mlp_clf.predict(X_trn_scaled))
            test_acc = metrics.accuracy_score(y_test, mlp_clf.predict(X_tst_scaled))
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model = mlp_clf
            print(f"\t\taccuracy={round(10000*test_acc) / 100}")
            config['test_acc'] = test_acc
            config['traing_acc'] = train_acc
            configs.append(config)
rcx = [
    ('hls', 'period_ms', 'num_periods'),
    ('period_ms', 'num_periods', 'hls'),
    ('num_periods', 'hls', 'period_ms'),
    ('num_periods', 'hls', 'total_history_ms'),
    ('period_ms', 'hls', 'total_history_ms'),
]
df = pd.DataFrame(configs)
for rows, cols, xs in rcx:
    # --------------------
    # Plot small multiples
    # --------------------
    g = sns.FacetGrid(data=df, col=cols, row=rows, ylim=(0.5,1), margin_titles=True)
    g.map(sns.lineplot, xs, 'test_acc', markers=True)
    plt.tight_layout()
    plt.savefig(f"img/test_accuracy vs r={rows},c={cols},x={xs}.png")
    plt.show()
    # ---------------------------
    # Plot pairwise relationships
    # ---------------------------
    g = sns.PairGrid(df[['test_acc', rows, cols, xs]], diag_sharey=False)
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot)
    plt.tight_layout()
    plt.savefig(f"img/pairwise r={rows},c={cols},x={xs}.png")
    plt.show()

print("Top 90% of models:")
best_models = df.loc[df.test_acc > df.test_acc.quantile(0.9), ['test_acc', 'period_ms', 'num_periods', 'total_history_ms', 'hls']].sort_values('test_acc', ascending=False)
print(best_models)

# --------------------------------------------------------
# Randomised Grid Search over the NN hyper-parameter space
# --------------------------------------------------------
hyper_params = []
scores = []
searches = []
pnh = []
from sklearn.model_selection import RandomizedSearchCV
distributions = {
        'tol': loguniform(1e-6, 1e-3),
        'alpha': loguniform(1e-7, 1e-3),
        'activation': ['tanh', 'relu', 'logistic'],
}

for i, (index, row) in enumerate(best_models.iterrows()):
    period_ms = row['period_ms']
    num_periods = row['num_periods']
    hls = row['hls']
    print(f"({i}/{len(best_models)}) Tuning model {hls=}, {period_ms=}, {num_periods=}")
    mlp = MLPClassifier(
            hidden_layer_sizes=hls,
            max_iter=5000,
            random_state=42,
            solver='adam')
    X_train, X_test, y_train, y_test = load_dataset(
            period_ms=period_ms, num_periods=num_periods
    )
    scaler = StandardScaler()
    _ = scaler.fit(X_train)
    X_trn_scaled = scaler.transform(X_train)
    clf = RandomizedSearchCV(
            mlp,
            distributions,
            random_state=42,
            n_iter=5,
            cv=4,
            verbose=3)
    search = clf.fit(X_trn_scaled, y_train)
    searches.append(search)
    pnh.append((period_ms, num_periods, hls))
    hyper_params.append(search.best_params_)
    scores.append(search.best_score_)


# ---------------------------------------------------------
# Find the best model, fit it, and plot a confusion matrix.
# ---------------------------------------------------------
best_params = df[df.test_acc == df.test_acc.max()]
X_train, X_test, y_train, y_test = load_dataset(
        period_ms=int(best_params.period_ms), num_periods=int(best_params.num_periods))
scaler = StandardScaler()
_ = scaler.fit(X_train) # Don't cheat - fit only on training data
X_tst_scaled = scaler.transform(X_test) # apply same transformation to test data
plot_conf_matrix(y_test,
        best_model.predict(X_tst_scaled),
        y_train, 
        f"MLP With {round(10000*float(best_params.test_acc))/100} test accuracy, {int(best_params.hls)} Hidden Nodes And {int(best_params.num_periods)} Period(s) of {int(best_params.period_ms)}ms Each")

# ======================================
# Attempt to train a Random Forest model
# ======================================
def train_rf():
    from sklearn.ensemble import RandomForestClassifier
    rf_clf = RandomForestClassifier(random_state=42)
    _ = rf_clf.fit(X_train, y_train)
    rf_score = rf_clf.score(X_test, y_test)
    print(rf_score)
    print(rf_clf.score(X_train, y_train))
    plot_conf_matrix(y_test, rf_clf.predict(X_test), y_train, f"Random Forest with score={round(rf_score * 10000) / 10000.0}")#, save_name="ConfusionMatrixRandomForest.png")

# =============================
# Attempt to train a KNN model
# =============================
def train_knn()
    best_n = 0
    best_score = 0.0
    for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100):
        clf = KNeighborsClassifier(n)
        _ = clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        if score > best_score:
            best_n = n
            best_score = score

    if best_n > 0:
        knn_clf= KNeighborsClassifier(best_n)
        _ = knn_clf.fit(X_train, y_train)
        print(f"Best knn is {best_n} with training: {knn_clf.score(X_train, y_train)}, and test: {knn_clf.score(X_test, y_test)}")

    plot_conf_matrix(y_test, knn_clf.predict(X_test), y_train, f"KNN with n={best_n} and test score: {knn_clf.score(X_test, y_test)}")

def plot_conf_matrix(y_test, y_pred, y_train, name, save_name=None):
    plt.rcParams.update({'font.size': 8})
    labels = pd.Series(y_train).value_counts().index
    full_labels = pd.Series(y_train).value_counts().reset_index().apply(
        lambda y: f"{y['index']} ({y[0]})", axis=1)
    print("Labels:")
    print(full_labels)
    cm = confusion_matrix(y_test, y_pred, labels=labels, normalize='true')
    plt.figure(figsize=(15,8))
    plt.title('Confusion matrix of ' + name)
    sns.heatmap(cm, xticklabels=labels, yticklabels=full_labels,
        annot=True, cmap="YlGnBu", fmt=".1g", linewidths=.5)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()
