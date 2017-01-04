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
nb_folds = 5

# loop over ass_assignments
dict_errors = defaultdict(list)
print("KFold cross-validation of the model on {} folds, assignment per assignment".format(nb_folds))
nb_assignments = len(DataLoader.LIST_ASS_ASSIGNMENTS)
for ass_assignment in DataLoader.LIST_ASS_ASSIGNMENTS:
    print("\t Cross validation Linex impact error on %s ... " % ass_assignment, end='', flush=True)
    # load data
    df_train = pd.read_hdf(train_path, key=ass_assignment)
    dates_train = df_train.index.values
    X_train = df_train.drop(["CSPL_RECEIVED_CALLS"], axis=1)
    X_train = feature_extractor.transform(X_train)
    y_train = df_train["CSPL_RECEIVED_CALLS"].values

    # cross validation
    list_errors_this_assignment = []
    kfold = KFold(nb_folds, random_state=random_state)
    for idFold, (train_indexes, test_indexes) in enumerate(kfold.split(X_train)):
        # TODO: Only use anterior dates for training during CV
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

    linex_error = float(np.mean(list_errors_this_assignment))
    print(
        "%0.4f" % (linex_error / nb_assignments)
    )
    # consolidated error for this ass_assignment
    if verbose >= 1:
        print(
            """\t Statistics on error:
            -> Mean: %0.2f
            -> Std: %0.2f
            -> Min: %0.2f
            -> Max: %0.2f
            -> Corresponding Linex Error: %0.2f""" % (float(np.mean(list_errors_this_assignment)),
                                                      float(np.std(list_errors_this_assignment)),
                                                      np.min(list_errors_this_assignment),
                                                      np.max(list_errors_this_assignment),
                                                      linex_error)
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
print("\nConsolidated error for each category")
linex_score = 0
for category, errors in dict_errors.items():
    nb_assignment_in_category = len(category)
    error = float(np.mean(errors))
    impact_on_linex_score = error * nb_assignment_in_category / nb_assignments
    print(
        """\t Statistics for category %s:
        -> Mean: %0.2f
        -> Std: %0.2f
        -> Min: %0.2f
        -> Max: %0.2f
        -> Impact of %s on final Linex score: %0.2f""" % (category,
                                                          float(np.mean(errors)) / nb_assignment_in_category,
                                                          float(np.std(errors)) / nb_assignment_in_category,
                                                          np.min(errors) / nb_assignment_in_category,
                                                          np.max(errors) / nb_assignment_in_category,
                                                          category,
                                                          impact_on_linex_score)
    )
    linex_score += impact_on_linex_score

print("\nWhich gives a final cross-validation Linex score of %0.3f" % linex_score)
