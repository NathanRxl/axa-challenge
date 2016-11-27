import pandas as pd
import time
import os


def create_submission(y_predict, x_test=pd.DataFrame(), folder="submissions"):
    """ Generate submission file from x_test and y_predict if x_test is given, else fill a submission with y_predict
    (y_predict must be sorted as in the submission files in this case)

    Parameters
    ----------
    x_test (facultative): type ndarray, shape (82909, >=2)
        Test dataset. Should contain at least two columns "DATE" and "ASS_ASSIGNMENT".
    y_predict: type ndarray, shape (82909, 1)
        Prediction done by ML. Should be an array of integers. If x_test is not provided, y_predict must be sorted in
        the same order than rows appear in the submisision file
    folder: type string
        folder in which the submission will be stored
    """
    if x_test.empty:
        df_submission = pd.read_csv("data/submission.txt", sep="\t")
    else:
        # create a dataframe that looks like the submission
        df_submission = pd.DataFrame()
        df_submission["DATE"] = x_test["DATE"]
        df_submission["ASS_ASSIGNMENT"] = x_test["ASS_ASSIGNMENT"]

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
