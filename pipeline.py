import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tools import DataLoader
from tools import Submissioner
import xgboost as xgb

train_path = "data/train.h5"
submission_path = "data/submission.h5"

# object that saves prediction and gather them into a final submission file
submissioner = Submissioner()
data_loader = DataLoader(train_path, submission_path)

# model
xgb_reg = Pipeline(
    [
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("estimator", xgb.XGBRegressor(n_estimators=150, learning_rate=0.8, nthread=4)),
    ]
)
total_time = 0
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

        # fit
        xgb_reg.fit(X_train, y_train)
        # predict
        y_predict = xgb_reg.predict(X_predict)

        # Performance note: the function save and auto_zeros_in_prediction take 3sec to run over the complete week loop
        # change the prediction to true 0 when the assignment is closed (or the prediction negative)
        submissioner.auto_zeros_in_prediction(y_predict, ass_assignment, X_predict)
        # save prediction
        submissioner.save(dates_predict, ass_assignment, y_predict)
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
