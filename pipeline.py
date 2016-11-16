from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tools import data_loader, create_submission

# parameters
path = "data/train_2011_2012_2013.csv"
nrows = 10000
randomState = 42

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

# fit
clf.fit(X_train, y_train)

# predict
y_predict = clf.predict(X_test)

# create submission
create_submission(X_test, y_predict)
