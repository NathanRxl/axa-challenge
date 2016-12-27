import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import KFold

from tools import DataLoader, Submissioner
from test_model import metrics

# import model
from model import AXARegressor, FeatureExtractorAXAReg, up_coef, DUST, DUST_2, FIVE_HUNDRED, BIG_BROTHERS, TELEPHONIE

feature_extractor = FeatureExtractorAXAReg()
clf = AXARegressor()

# parameters
train_path = "data/train.h5"
verbose = 0
random_state = 42
nb_folds = 10

# loop over ass_assignments
dict_errors = defaultdict(list)
for ass_assignment in DataLoader.LIST_ASS_ASSIGNMENTS:
    print("%s..." % ass_assignment)
    # load data
    df_train = pd.read_hdf(train_path, key=ass_assignment)
    dates_train = df_train.index.values
    X_train = df_train.drop(["CSPL_RECEIVED_CALLS"], axis=1)
    y_train = df_train["CSPL_RECEIVED_CALLS"].values

    # cross validation
    list_errors_this_assignment = []
    kfold = KFold(nb_folds, random_state=random_state)
    for idFold, (train_indexes, test_indexes) in enumerate(kfold.split(X_train)):
        X_train_CV = X_train.loc[train_indexes]
        y_train_CV = y_train[train_indexes]
        X_train_CV = feature_extractor.transform(X_train_CV)

        X_test_CV = X_train.loc[test_indexes]
        y_test_CV = y_train[test_indexes]
        X_test_CV = feature_extractor.transform(X_test_CV)

        # fit
        clf.fit(X_train_CV, y_train_CV, ass_assignment)

        # predict
        y_predict_CV = clf.predict(X_test_CV, ass_assignment)
        submissioner = Submissioner()
        y_predict_CV = submissioner.up_prediction(y_predict_CV, ass_assignment, up_coef)
        y_predict_CV = submissioner.auto_zeros_in_prediction(y_predict_CV, ass_assignment, X_test_CV)

        # compute error
        error = metrics.linex_score(y_test_CV, y_predict_CV)
        list_errors_this_assignment.append(error)
        if verbose >= 1:
            print("KFold #%d:" % idFold)
            print("LinEx error for this fold: %0.2f\n" % error)

    # consolidated error for this ass_assignment
    if verbose >= 1:
        print(
            """
            Statistics on error:
                \n   -> Mean: %0.2f
                \n   -> Std: %0.2f
                \n   -> Min: %0.2f
                \n   -> Max: %0.2f

            \n""" % (np.mean(list_errors_this_assignment), np.std(list_errors_this_assignment), np.min(list_errors_this_assignment), np.max(list_errors_this_assignment))
        )

    # keep tracks of all errors
    if ass_assignment in DUST:
        dict_errors["DUST"].extend(list_errors_this_assignment)
    elif ass_assignment in DUST_2:
        dict_errors["DUST_2"].extend(list_errors_this_assignment)
    elif ass_assignment in FIVE_HUNDRED:
        dict_errors["FIVE_HUNDRED"].extend(list_errors_this_assignment)
    elif ass_assignment in BIG_BROTHERS:
        dict_errors["BIG_BROTHERS"].extend(list_errors_this_assignment)
    elif ass_assignment in TELEPHONIE:
        dict_errors["TELEPHONIE"].extend(list_errors_this_assignment)
    else:
        raise ValueError

# consolidated error for each category of ass_assignment
print("Consolidated error for each category")
for category, errors in dict_errors.items():
    print(
        """
        Statistics for category %s:
            \n   -> Mean: %0.2f
            \n   -> Std: %0.2f
            \n   -> Min: %0.2f
            \n   -> Max: %0.2f

        \n""" % (category, np.mean(errors), np.std(errors), np.min(errors), np.max(errors))
    )
