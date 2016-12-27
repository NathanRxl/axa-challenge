import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from tools import DataLoader


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
        for ass_assignment in DataLoader.LIST_ASS_ASSIGNMENTS:
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
                raise ('ASS_ASSIGNMENT {} unknown.'.format(ass_assignment))

    def fit(self, X_transform, y, assignment):
        self.dict_reg_xgb[assignment].fit(X_transform, y)

    def predict(self, X_transform, assignment):
        return self.dict_reg_xgb[assignment].predict(X_transform)
