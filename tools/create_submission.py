import pandas as pd
import time


def create_submission(X_test, y_predict):
    """ Generate submission file from X_test and y_predict """

    # create a dataframe that looks like the submission
    df_submission = pd.DataFrame()
    df_submission["DATE"] = X_test["DATE"]
    df_submission["ASS_ASSIGNMENT"] = X_test["ASS_ASSIGNMENT"]
    df_submission["prediction"] = y_predict

    # convert dataframe to string
    submission_content = df_submission.to_csv(index=False,date_format="%Y-%m-%d %H:%M:%S.000",sep="\t")

    file_name = "submission_%s.txt" % time.strftime("%Y%m%d_%H%M%S")
    with open(file_name, 'w') as file:
        file.write(submission_content)