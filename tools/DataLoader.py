import pandas as pd


def data_loader(path, nrows):
    # load row dataset from csv file
    row_dataset = pd.read_csv(path, nrows=nrows, sep=";")

    # TODO: retrieve X_train, y_train and X_test from row_dataset
    X_train = 0
    X_test = 0
    y_train = 0

    return X_train, y_train, X_test


class DataLoader:
    WEEKS_DATES = {
        0 : (pd.Timestamp("2012-12-28 00:00:00"), pd.Timestamp('2013-01-03 23:30:00')),
        1 : (pd.Timestamp('2013-02-02 00:00:00'), pd.Timestamp('2013-02-08 23:30:00')),
        2 : (pd.Timestamp('2013-03-06 00:00:00'), pd.Timestamp('2013-03-12 23:30:00')),
        3 : (pd.Timestamp('2013-04-10 00:00:00'), pd.Timestamp('2013-04-16 23:30:00')),
        4 : (pd.Timestamp('2013-05-13 00:30:00'), pd.Timestamp('2013-05-19 23:30:00')),
        5 : (pd.Timestamp('2013-06-12 00:00:00'), pd.Timestamp('2013-06-18 23:30:00')),
        6 : (pd.Timestamp('2013-07-16 00:00:00'), pd.Timestamp('2013-07-22 23:30:00')),
        7 : (pd.Timestamp('2013-08-15 00:00:00'), pd.Timestamp('2013-08-21 23:30:00')),
        8 : (pd.Timestamp('2013-09-14 00:00:00'), pd.Timestamp('2013-09-20 23:30:00')),
        9 : (pd.Timestamp('2013-10-18 00:00:00'), pd.Timestamp('2013-10-24 23:30:00')),
        10 : (pd.Timestamp('2013-11-20 00:00:00'), pd.Timestamp('2013-11-26 23:30:00')),
        11 : (pd.Timestamp('2013-12-22 00:00:00'), pd.Timestamp('2013-12-28 23:30:00'))
    }

    LIST_ASS_ASSIGNMENTS = ['CMS', 'Crises', 'Domicile', 'Gestion',
                            'Gestion - Accueil Telephonique', 'Gestion Assurances',
                            'Gestion Relation Clienteles', 'Gestion Renault', 'Japon',
                            'Médical', 'Nuit', 'RENAULT', 'Regulation Medicale', 'SAP',
                            'Services', 'Tech. Axa', 'Tech. Inter', 'Téléphonie', 'Tech. Total',
                            'Mécanicien', 'CAT', 'Manager', 'Gestion Clients', 'Gestion DZ',
                            'RTC', 'Prestataires']

    def __init__(self, train_path, submission_path):
        self.train_path = train_path
        self.submission_path = submission_path

        self.week_nb = None
        self.week_dates = None
        self.idx_ass_assignment = None

    def update_week(self, week_nb):
        # update week and retrieve dates of the beginning and the end of the week
        self.week_nb = week_nb
        self.week_begin = DataLoader.WEEKS_DATES[week_nb][0]
        self.week_end = DataLoader.WEEKS_DATES[week_nb][1]

        # reinitialize idx_ass_assignment
        self.idx_ass_assignment = 0

    def __iter__(self):
        return self

    def next(self):
        try:
            # for each iteration we retrieve the ass_assignment
            ass_assignment = DataLoader.LIST_ASS_ASSIGNMENTS[self.idx_ass_assignment]
        except IndexError:
            # if a IndexError is raised, it means that we reached the end of the list of ass_assignments
            # we then stop the loop
            raise StopIteration

        # read hdf files
        train_df = pd.read_hdf(self.train_path, key=ass_assignment)
        submission_df = pd.read_hdf(self.submission_path, key=ass_assignment)

        # split data
        dates_train = train_df["DATE"]
        dates_test = submission_df["DATE"]
        X_train = train_df.drop(["DATE", "CSPL_RECEIVED_CALLS"], axis=1)
        X_test = submission_df.drop(["DATE", "prediction"], axis=1)
        y_train = train_df["CSPL_RECEIVED_CALLS"].values

        self.idx_ass_assignment += 1
        return ass_assignment, dates_train, X_train, y_train, dates_test, X_test
