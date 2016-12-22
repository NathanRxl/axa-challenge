import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from tools import DataLoader, Submissioner

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

# 23 Max call during max(Daytime, Night) -> leads in average to predict 0 or 1 per 30 min
DUST = {
    'Prestataires': 2,
    'CMS': 3,
    'Gestion': 3,
    'Gestion Renault': 3,
    'Gestion DZ': 4,
    'Manager': 6,
    'Gestion Relation Clienteles': 7,
    'Gestion Clients': 7,
    'Mécanicien': 10,
    'Gestion Assurances': 12,
    'Regulation Medicale': 12,
    'Japon': 14,
    'SAP': 15,
    'RTC': 23,
}
# Between 39 and 72 Max call during max(Daytime, Night) -> leads in average to predict 0, 1, 2 or 3 per 30 min
DUST_2 = {
    'Crises': 39,
    'Médical': 52,
    'Domicile': 68,
    'Gestion - Accueil Telephonique': 72,
}

# Between 100 Max call during max(Daytime, Night) -> begin to have a significant impact on score if mispredicted
FIVE_HUNDRED = {
    'RENAULT': 100,
    'Services': 100,
    'Tech. Inter': 100,
    'Tech. Total': 100,
    'Nuit': 110,
}

# Major impact on score :
BIG_BROTHERS = {
    'CAT': 250,
    'Tech. Axa': 410,
}

# Impact on score is similar to the one of all the others together
TELEPHONIE = {
    'Téléphonie': 1380,
}

shared_nthread = -1
shared_objective = 'reg:linear'

XGB_params = {
    'DUST': {
        'n_estimators': 25,
        'max_depth': 4,
        'learning_rate': 0.9,
        'nthread': shared_nthread,
        'objective': shared_objective,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
    },

    'DUST_2': {
        'n_estimators': 45,
        'max_depth': 4,
        'learning_rate': 0.9,
        'nthread': shared_nthread,
        'objective': shared_objective,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
    },

    'FIVE_HUNDRED': {
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.8,
        'nthread': shared_nthread,
        'objective': shared_objective,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
    },

    'BIG_BROTHERS': {
        'n_estimators': 175,
        'max_depth': 4,
        'learning_rate': 0.7,
        'nthread': shared_nthread,
        'objective': shared_objective,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
    },

    'TELEPHONIE': {
        'n_estimators': 850,
        'max_depth': 5,
        'learning_rate': 0.8,
        'nthread': shared_nthread,
        'objective': shared_objective,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
    },
}

up_coef = {
    # TELEPHONIE
    'Téléphonie': 1.10,
    # BIG_BROTHERS
    'Tech. Axa': 1.40,
    'CAT': 1.40,
    # FIVE_HUNDRED
    'Nuit': 1.0,
    'RENAULT': 1.20,
    'Services': 1.20,
    'Tech. Inter': 1.20,
    'Tech. Total': 1.20,
    # DUST_2
    'Gestion - Accueil Telephonique': 1.10,
    'Domicile': 1.10,
    'Médical': 1.10,
    'Crises': 1.10,
    # DUST
    'RTC': 1.10,
    'SAP': 1.10,
    'Japon': 1.10,
    'Gestion Assurances': 1.10,
    'Regulation Medicale': 1.10,
    'Mécanicien': 1.10,
    'Gestion Relation Clienteles': 1.10,
    'Gestion Clients': 1.10,
    'Manager': 1.10,
    'Gestion DZ': 1.10,
    'CMS': 1.10,
    'Gestion': 1.10,
    'Gestion Renault': 1.10,
    'Prestataires': 1.10,
}


class AXARegressor(BaseEstimator):
    def __init__(self):
        self.dict_reg_xgb = dict()
        for ass_assignment in data_loader.LIST_ASS_ASSIGNMENTS:
            if ass_assignment in DUST:
                self.dict_reg_xgb[ass_assignment] = Pipeline(
                    [
                        ("scaler", StandardScaler(with_mean=True, with_std=True)),
                        ("reg", xgb.XGBRegressor(**XGB_params['DUST'])),
                    ]
                )

            elif ass_assignment in DUST_2:
                self.dict_reg_xgb[ass_assignment] = Pipeline(
                    [
                        ("scaler", StandardScaler(with_mean=True, with_std=True)),
                        ("reg", xgb.XGBRegressor(**XGB_params['DUST_2'])),
                    ]
                )

            elif ass_assignment in FIVE_HUNDRED:
                self.dict_reg_xgb[ass_assignment] = Pipeline(
                    [
                        ("scaler", StandardScaler(with_mean=True, with_std=True)),
                        ("reg", xgb.XGBRegressor(**XGB_params['FIVE_HUNDRED'])),
                    ]
                )

            elif ass_assignment in BIG_BROTHERS:
                self.dict_reg_xgb[ass_assignment] = Pipeline(
                    [
                        ("scaler", StandardScaler(with_mean=True, with_std=True)),
                        ("reg", xgb.XGBRegressor(**XGB_params['BIG_BROTHERS'])),
                    ]
                )

            elif ass_assignment in TELEPHONIE:
                self.dict_reg_xgb[ass_assignment] = Pipeline(
                    [
                        ("scaler", StandardScaler(with_mean=True, with_std=True)),
                        ("reg", xgb.XGBRegressor(**XGB_params['TELEPHONIE'])),
                    ]
                )
            else:
                raise('ASS_ASSIGNMENT {} unknown.'.format(ass_assignment))

    def fit(self, X_transform, y, assignment):
        self.dict_reg_xgb[assignment].fit(X_transform, y)

    def predict(self, X_transform, assignment):
        return self.dict_reg_xgb[assignment].predict(X_transform)

feature_extractor_reg = FeatureExtractorAXAReg()
xgb_reg = AXARegressor()
# Performance note: use anterior predictions add 4 sec to the complete week loop
use_anterior_predictions = False
hybrid_approach = False
plot_feature_importance = False

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

        # if you want to exclude some assignment from the prediction uncomment here
        # if ass_assignment in TELEPHONIE:
        #     continue
        # if ass_assignment in BIG_BROTHERS:
        #     continue
        # if ass_assignment in FIVE_HUNDRED:
        #     continue
        # if ass_assignment in DUST:
        #     continue
        # if ass_assignment in DUST_2:
        #     continue

        if use_anterior_predictions:
            # debug checks: X_train length is augmented by Xtra_train length
            initial_X_train_len = len(X_train)
            X_train, y_train = feature_extractor_reg.fit(X_train, y_train, Xtra_train[ass_assignment])
            # debug checks: X_train length is augmented by Xtra_train length
            if Xtra_train[ass_assignment] is not None:
                assert len(Xtra_train[ass_assignment]) + initial_X_train_len == len(X_train)

        # fit
        X_train = feature_extractor_reg.transform(X_train)

        # predict
        X_predict = feature_extractor_reg.transform(X_predict)

        # You can put the assignments you want to predict with the heuristic in this if
        if hybrid_approach and (ass_assignment in BIG_BROTHERS or ass_assignment in TELEPHONIE):
            """ Use the best heuristic we found for now """
            X_train["CSPL_RECEIVED_CALLS"] = y_train
            y_predict = list()
            prediction = X_train.groupby(["YEAR", "HOUR", "NOT_WORKING_DAY"])["CSPL_RECEIVED_CALLS"].max().to_dict()
            for year, hour, not_working_day in X_predict[["YEAR", "HOUR", "NOT_WORKING_DAY"]].values:
                try:
                    y_predict.append(prediction[(year, hour, not_working_day)])
                except KeyError:
                    initial_len_predict = len(y_predict)
                    y, h = 0, 0
                    while initial_len_predict == len(y_predict):
                        try:
                            if h == 0:
                                h += 1
                            elif h > 0:
                                h = -h
                            elif h < 0:
                                h = -h + 1
                            elif h > 2:
                                y += 1
                                h = 0
                            y_predict.append(prediction[(year - y, hour - h, not_working_day)])
                        except KeyError:
                            continue
            y_predict = np.array(y_predict)
            X_train = X_train.drop(["CSPL_RECEIVED_CALLS"], axis=1)
        else:
            xgb_reg.fit(X_train, y_train, ass_assignment)
            if plot_feature_importance and (ass_assignment in BIG_BROTHERS or ass_assignment in TELEPHONIE):
                xgb.plot_importance(xgb_reg.dict_reg_xgb[ass_assignment].named_steps['reg'])
                plt.title("{}".format(ass_assignment))
                plt.show()
            y_predict = xgb_reg.predict(X_predict, ass_assignment)

        # When predictions are negative make them all positive by increasing them all,
        # then apply a coefficient assignment specific
        y_predict = submissioner.up_prediction(y_predict, ass_assignment, up_coef)
        # Performance note: the function save and auto_zeros_in_prediction take 3sec to run over the complete week loop
        # change the prediction to true 0 when the assignment is closed (or the prediction negative)
        y_predict = submissioner.auto_zeros_in_prediction(y_predict, ass_assignment, X_predict)
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
# in order to create a submission from a particular anterior submission
submissioner.create_submission(ref_submission_path="submissions/submission76.txt")
# submissioner.create_submission()
print("\nSubmission added in submissions/ !")
