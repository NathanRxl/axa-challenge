import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from tools import DataLoader
from tools import Submissioner
import xgboost as xgb

train_path = "data/train.h5"
submission_path = "data/submission.h5"

# object that saves prediction and gather them into a final submission file
submissioner = Submissioner()
data_loader = DataLoader(train_path, submission_path)


class FeatureExtractorAXAReg:
    def __init__(self):
        pass

    def fit(self, X_df, y, Xtra_df):
        if Xtra_df is None:
            return X_df, y
        else:
            X_df["CSPL_RECEIVED_CALLS"] = y
            X_df = pd.concat([X_df, Xtra_df])
            X_df.sort_index(axis=0, inplace=True, ascending=True)
            y = X_df["CSPL_RECEIVED_CALLS"].values
            X_df = X_df.reset_index(drop=True).drop(["CSPL_RECEIVED_CALLS"], axis=1)
            return X_df, y

    def transform(self, X_df):
        return X_df.reset_index(drop=True)


class AXARegressor(BaseEstimator):
    def __init__(self, n_estimators=150, learning_rate=0.8, nthread=4):
        self.dict_reg_xgb = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("reg", xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, nthread=nthread)),
            ]
        )

    def fit(self, X_transform, y):
        self.dict_reg_xgb.fit(X_transform, y)

    def predict(self, X_transform):
        return self.dict_reg_xgb.predict(X_transform)

feature_extractor_reg = FeatureExtractorAXAReg()
xgb_reg = AXARegressor()
# Performance note: use anterior predictions add 4 sec to the complete week loop
use_anterior_predictions = True

if use_anterior_predictions:
    Xtra_train = dict()
    for ass_assignment in data_loader.LIST_ASS_ASSIGNMENTS:
        Xtra_train[ass_assignment] = None

# Performance note: the following loop takes 4 sec to load the data over all the 12 weeks to predict
# loop over week of submissions
for week_nb in np.arange(12):
    print("Prediction for week #(%d/11) ... " % week_nb, end="", flush=True)
    data_loader.update_week(week_nb)

    # loop over ass_assignments
    for ass_assignment, dates_train, X_train, y_train, dates_predict, X_predict in data_loader:
        # sometimes we don't need to do predictions for certain week and ass_assignment, we then continue the loop
        # without processing ML
        if len(dates_predict) == 0:
            continue

        if use_anterior_predictions:
            # debug checks: X_train length is augmented by Xtra_train length
            initial_X_train_len = len(X_train)
            X_train, y_train = feature_extractor_reg.fit(X_train, y_train, Xtra_train[ass_assignment])
            # debug checks: X_train length is augmented by Xtra_train length
            if Xtra_train[ass_assignment] is not None:
                assert len(Xtra_train[ass_assignment]) + initial_X_train_len == len(X_train)

        # fit
        X_train = feature_extractor_reg.transform(X_train)
        xgb_reg.fit(X_train, y_train)

        # predict
        X_predict = feature_extractor_reg.transform(X_predict)
        y_predict = xgb_reg.predict(X_predict)

        # Performance note: the function save and auto_zeros_in_prediction take 3sec to run over the complete week loop
        # change the prediction to true 0 when the assignment is closed (or the prediction negative)
        submissioner.auto_zeros_in_prediction(y_predict, ass_assignment, X_predict)
        # save prediction
        submissioner.save(dates_predict, ass_assignment, y_predict)

        if use_anterior_predictions:
            # debug checks: Xtra_train[ass_assignment] length is increasing
            if Xtra_train[ass_assignment] is not None:
                initial_Xtra_train_len = len(Xtra_train[ass_assignment])
            else:
                initial_Xtra_train_len = 0

            # store predictions into an extra train set for future predictions
            X_predict["CSPL_RECEIVED_CALLS"] = y_predict
            X_predict.set_index(pd.DatetimeIndex(dates_predict), inplace=True)
            Xtra_train[ass_assignment] = pd.concat([Xtra_train[ass_assignment], X_predict])

            # debug checks: Xtra_train[ass_assignment] length is increasing
            if Xtra_train[ass_assignment] is not None:
                assert len(Xtra_train[ass_assignment]) >= initial_Xtra_train_len
    print("OK")

if submissioner.nb_negative_predictions > 0:
    print(
        "WARN: auto_zeros_in_prediction corrected",
        "{}/{}".format(submissioner.nb_negative_predictions, submissioner.SUBMISSION_TOTAL_LENGTH),
        "negative predictions for this submission"
    )
print("INFO: auto_zeros_in_prediction improved the score by:", submissioner.auto_zeros_impact, "for this submission")
submissioner.create_submission()

print("\nSubmission added in submissions/ !")
