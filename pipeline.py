import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xgboost as xgb

from tools import DataLoader, Submissioner
from model import AXARegressor, FeatureExtractorAXAReg, up_coef, DUST, DUST_2, FIVE_HUNDRED, BIG_BROTHERS, TELEPHONIE

train_path = "data/train.h5"
submission_path = "data/submission.h5"

# object that saves prediction and gather them into a final submission file
submissioner = Submissioner()
data_loader = DataLoader(train_path, submission_path)

# retrieve model
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
submissioner.create_submission()
# submissioner.create_submission(ref_submission_path="submissions/submission76.txt")
print("\nSubmission added in submissions/ !")
