import pandas as pd
import numpy as np
import time
import os


class Submissioner:
    def __init__(self):
        self.predictions = list()
        # during the saving process, we will save a multi-index created from dates and ass_assignments
        self.multi_index = None

    def create_submission(self, ref_submission_path="data/submission.txt", folder="submissions"):
        # retrieve submission file reference
        ref_submission = pd.read_csv(ref_submission_path, sep='\t')
        ref_submission["DATE"] = ref_submission["DATE"].apply(lambda date: pd.Timestamp(date))

        # set predictions in dataframe
        ref_submission.set_index(["DATE", "ASS_ASSIGNMENT"], inplace=True)
        ref_submission.loc[self.multi_index] = np.array(self.predictions).reshape((len(self.predictions), 1))
        submission = ref_submission.reset_index()

        # convert dataframe to string
        submission_content = submission.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S.000", sep="\t")

        # create folder to store submission file if the folder doesn't exist yet
        if not os.path.exists(folder):
            os.makedirs(folder)

        # create file name according to current date
        file_name = "%s/submission_%s.txt" % (folder, time.strftime("%Y%m%d_%H%M%S"))

        # write into submission file
        with open(file_name, 'w') as file:
            file.write(submission_content)

    def save(self, dates_test, ass_assignment, y_predict):
        if self.multi_index is None:
            self.multi_index = pd.MultiIndex.from_product([dates_test, [ass_assignment]])
        else:
            self.multi_index = self.multi_index.append(pd.MultiIndex.from_product([dates_test, [ass_assignment]]))
        self.predictions.extend(y_predict)
