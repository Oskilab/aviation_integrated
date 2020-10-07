from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from IPython import embed
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
# TODO: regenerate ml_dataset (need to do same processing as in combine.py

results = pd.read_csv('results/ml_dataset.csv', index_col = 0)
y_cols = ['ntsb_accidents', 'ntsb_incidents', 'faa_incidents']

drop_cols = ['tracon_month', 'year']
for idx, dtype in enumerate(results.dtypes):
    if results.dtypes.iloc[idx] == 'object':
        colname = results.columns[idx]
        try:
            results[colname] = results[colname].str.replace(',', '').astype(float)
        except ValueError:
            pass

for y_col in y_cols:
    results[y_col] = results[y_col].fillna(0)

for idx, dtype in enumerate(results.dtypes):
    if results.dtypes.iloc[idx] == 'object':
        drop_cols.append(results.columns[idx])

num_na = results.isna().sum() / results.shape[0]

keep_these_cols = results.columns[(num_na > 0) & (num_na < 0.5)]
for col in keep_these_cols:
    if results.dtypes[col] == 'float64':
        avg_nonna = results.loc[~results[col].isna(), col].mean()
        results[col] = results[col].fillna(avg_nonna)
        if col in drop_cols:
            drop_cols.remove(col)

used_columns = np.array(results.drop(y_cols + drop_cols + list(results.columns[num_na > 0]), axis = 1).columns)
x = results.drop(y_cols + drop_cols + list(results.columns[num_na > 0]), axis = 1).values
y = results['ntsb_accidents'].values

y_onehot = OneHotEncoder(handle_unknown = 'ignore').fit_transform(y.reshape(-1, 1)).todense()

x_train, x_test, y_train, y_test = train_test_split(x, y_onehot, test_size = 0.2, random_state = 42)
y_train = np.squeeze(np.asarray(y_train))
y_test = np.squeeze(np.asarray(y_test))

rfr = RandomForestClassifier(oob_score = True, verbose = True, n_estimators = 300, class_weight = 'balanced_subsample')
rfr.fit(x_train, y_train)

print(confusion_matrix(y_test.argmax(axis = 1), rfr.predict(x_test).argmax(axis = 1)))
embed()
print(used_columns)
