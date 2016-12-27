import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from tools import data_loader
from test_model import linex_error

# parameters
path = "data/train_2011_2012_2013.csv"
nrows = 10000
verbose = 1
randomState = 42
nbFolds = 10

# load data
# TODO: create h5 file from csv with cleaned data then load data directly from h5 file
X_train, y_train, X_test = data_loader(path, nrows)

# classifier
clf = Pipeline(
    [
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("estimator", RandomForestRegressor(n_estimators=300, max_depth=15, n_jobs=-1, random_state=randomState)),
    ]
)

# cross-validation
list_errors = []
for idFold, (train_indexes, test_indexes) in enumerate(KFold(len(X_train), nbFolds, random_state=randomState)):
    X_train_CV = X_train.loc[train_indexes]
    y_train_CV = y_train[train_indexes]

    X_test_CV = X_train.loc[test_indexes]
    y_test_CV = y_train[test_indexes]

    # fit
    clf.fit(X_train_CV, y_train_CV)

    # predict
    y_predict_CV = clf.predict(X_test_CV)

    # compute error
    error = linex_error(y_test_CV, y_predict_CV)
    list_errors.append(error)
    if verbose >= 1:
        print("KFold #%d:" % idFold)
        print("LinEx error for this fold: %0.2f\n" % error)

# consolidated error
if verbose >= 1:
    print(
        """
        Statistics on error:
            \n   -> Mean: %0.2f
            \n   -> Std: %0.2f
            \n   -> Min: %0.2f
            \n   -> Max: %0.2f

        """ % (np.mean(list_errors), np.std(list_errors), np.min(list_errors), np.max(list_errors))
    )
