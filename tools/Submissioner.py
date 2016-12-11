import pandas as pd
import numpy as np
import time
import os
from test_model import metrics


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

    SUBMISSION_TOTAL_LENGTH = 82909

    auto_zeros_impact = 0.0

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

    def auto_zeros_in_prediction(self, y_predict, ass_assignment, X_predict, verbose=0):
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
            index_to_change = set(index_to_change)
            total_change_in_loss = 0.0
            nb_index_changed = 0

            for index in index_to_change:
                local_change_in_loss = metrics.linex_score([0.0], [y_predict[index]])
                if local_change_in_loss < 0.0:
                    total_change_in_loss += local_change_in_loss
                    y_predict[index] = 0.0
                    if verbose == 2:
                        print(
                            "Index", index, "of prediction for assignment", ass_assignment,
                            "was changed from", y_predict[index], "to 0.0"
                        )
                    if verbose >= 1:
                        nb_index_changed += 1

            if verbose >= 1:
                if total_change_in_loss < 0.0:
                    print(
                        "auto_zeros_function changed", nb_index_changed, "indexes of the prediction, leading to a",
                        "global improvement of the prediction score of",
                        - total_change_in_loss / float(self.SUBMISSION_TOTAL_LENGTH)
                    )

            self.auto_zeros_impact += - total_change_in_loss / float(self.SUBMISSION_TOTAL_LENGTH)

    def save(self, dates_test, ass_assignment, y_predict):
        if self.multi_index is None:
            self.multi_index = pd.MultiIndex.from_product([dates_test, [ass_assignment]])
        else:
            self.multi_index = self.multi_index.append(pd.MultiIndex.from_product([dates_test, [ass_assignment]]))
        self.predictions.extend(y_predict)
