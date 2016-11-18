import pandas as pd
import time
import os


def create_submission(X_test, y_predict, folder="submissions"):
    """ Generate submission file from X_test and y_predict

    :param X_test: test dataset. Should contain at least two columns "DATE" and "ASS_ASSIGNMENT"
    :param y_predict: prediction done by ML. Should be an array of integers and have the same length than X_test
    :param folder: folder in which the submission will be stored
    """

    # create a dataframe that looks like the submission
    df_submission = pd.DataFrame()
    df_submission["DATE"] = X_test["DATE"]
    df_submission["ASS_ASSIGNMENT"] = X_test["ASS_ASSIGNMENT"]
    df_submission["prediction"] = y_predict.astype(int)  # in case we forgot to convert in int before

    # convert dataframe to string
    submission_content = df_submission.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S.000", sep="\t")

    # create folder to store submission file if the folder doesn't exist yet
    if not os.path.exists(folder):
        os.makedirs(folder)

    # create file name according to current date
    file_name = "%s/submission_%s.txt" % (folder, time.strftime("%Y%m%d_%H%M%S"))

    # write into submission file
    with open(file_name, 'w') as file:
        file.write(submission_content)
