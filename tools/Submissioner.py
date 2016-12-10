import pandas as pd
import numpy as np
import time
import os


class Submissioner:

    ASSIGNMENTS_CLOSED_OVER_NIGHT = [
        'CMS',
        'Gestion Amex',
        'Gestion Assurances',
        'Gestion Clients',
        'Gestion DZ',
        'Gestion Relation Clienteles',
        'Gestion Renault',
        'Prestataires'
    ]

    ASSIGNMENTS_CLOSED_DURING_DAYOFFS = [
        'CMS',
        'Crises',
        'Gestion',
        'Gestion Amex',
        'Gestion Assurances',
        'Gestion Clients',
        'Gestion DZ',
        'Gestion Relation Clienteles',
        'Gestion Renault',
        'Prestataires',
        'RTC'
    ]

    ASSIGNMENTS_CLOSED_DURING_NOT_WORKING_DAYS = [
        'CMS',
        'Gestion',
        'Gestion Amex',
        'Gestion Clients',
        'Gestion DZ',
        'Gestion Relation Clienteles',
        'Gestion Renault',
        'Prestataires'
     ]

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

    def auto_zeros_in_prediction(self, y_predict, ass_assignment, X_predict):
        """
        Some assignments seem closed during night or during day offs.
        This method uses this information to put call prediction to 0 during these periods.
        /!\ Therefore, this method modify the prediction
        """
        index_to_change = list()

        if ass_assignment in self.ASSIGNMENTS_CLOSED_OVER_NIGHT:
            index_to_change.extend(X_predict[X_predict["DAYTIME"] == 0].index.tolist())
        if ass_assignment in self.ASSIGNMENTS_CLOSED_DURING_DAYOFFS:
            index_to_change.extend(X_predict[X_predict["DAY_OFF"] == 1].index.tolist())
        if ass_assignment in self.ASSIGNMENTS_CLOSED_DURING_NOT_WORKING_DAYS:
            index_to_change.extend(X_predict[X_predict["NOT_WORKING_DAY"] == 1].index.tolist())

        if index_to_change:
            for index in set(index_to_change):
                y_predict[index] = 0.0

    def save(self, dates_test, ass_assignment, y_predict):
        if self.multi_index is None:
            self.multi_index = pd.MultiIndex.from_product([dates_test, [ass_assignment]])
        else:
            self.multi_index = self.multi_index.append(pd.MultiIndex.from_product([dates_test, [ass_assignment]]))
        self.predictions.extend(y_predict)
