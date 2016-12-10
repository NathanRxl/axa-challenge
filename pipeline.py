import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tools import DataLoader
from tools import Submissioner

train_path = "data/train.h5"
submission_path = "data/submission.h5"

# object that saves prediction and gather them into a final submission file
submissioner = Submissioner()
data_loader = DataLoader(train_path, submission_path)

# loop over week of submissions
for week_nb in np.arange(12):
    print("\nWeek #%d..." % week_nb)
    data_loader.update_week(week_nb)

    # loop over ass_assignments
    for ass_assignment, dates_train, X_train, y_train, dates_predict, X_predict in data_loader:

        X_predict = X_predict.reset_index(drop=True)

        # sometimes we don't need to do predictions for certain week and ass_assignment, we then continue the loop without processing ML
        if len(dates_predict) == 0:
            continue

        # model
        clf = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("estimator", RandomForestRegressor(n_estimators=10, max_depth=5, n_jobs=-1, random_state=42)),
            ]
        )

        # fit
        clf.fit(X_train, y_train)

        # predict
        y_predict = clf.predict(X_predict)
        submissioner.auto_zeros_in_prediction(y_predict, ass_assignment, X_predict)

        # save prediction
        submissioner.save(dates_predict, ass_assignment, y_predict)

submissioner.create_submission()

print("\nDone!")
