import pandas as pd


class DataLoader:
    WEEKS_DATES = {
        0: (pd.Timestamp("2012-12-28 00:00:00"), pd.Timestamp('2013-01-03 23:30:00')),
        1: (pd.Timestamp('2013-02-02 00:00:00'), pd.Timestamp('2013-02-08 23:30:00')),
        2: (pd.Timestamp('2013-03-06 00:00:00'), pd.Timestamp('2013-03-12 23:30:00')),
        3: (pd.Timestamp('2013-04-10 00:00:00'), pd.Timestamp('2013-04-16 23:30:00')),
        4: (pd.Timestamp('2013-05-13 00:30:00'), pd.Timestamp('2013-05-19 23:30:00')),
        5: (pd.Timestamp('2013-06-12 00:00:00'), pd.Timestamp('2013-06-18 23:30:00')),
        6: (pd.Timestamp('2013-07-16 00:00:00'), pd.Timestamp('2013-07-22 23:30:00')),
        7: (pd.Timestamp('2013-08-15 00:00:00'), pd.Timestamp('2013-08-21 23:30:00')),
        8: (pd.Timestamp('2013-09-14 00:00:00'), pd.Timestamp('2013-09-20 23:30:00')),
        9: (pd.Timestamp('2013-10-18 00:00:00'), pd.Timestamp('2013-10-24 23:30:00')),
        10: (pd.Timestamp('2013-11-20 00:00:00'), pd.Timestamp('2013-11-26 23:30:00')),
        11: (pd.Timestamp('2013-12-22 00:00:00'), pd.Timestamp('2013-12-28 23:30:00'))
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
        self.train_df = dict()
        self.submission_df = dict()
        for ass_assignment in self.LIST_ASS_ASSIGNMENTS:
            self.train_df[ass_assignment] = pd.read_hdf(self.train_path, key=ass_assignment)
            self.submission_df[ass_assignment] = pd.read_hdf(self.submission_path, key=ass_assignment)

    def update_week(self, week_nb):
        # update week and retrieve dates of the beginning and the end of the week
        self.week_nb = week_nb
        self.week_begin = DataLoader.WEEKS_DATES[week_nb][0]
        self.week_end = DataLoader.WEEKS_DATES[week_nb][1]

        # reinitialize idx_ass_assignment
        self.idx_ass_assignment = -1

    def __iter__(self):
        return self

    def __next__(self):
        try:
            # update idx_ass_assignment
            self.idx_ass_assignment += 1

            # for each iteration we retrieve the ass_assignment
            ass_assignment = DataLoader.LIST_ASS_ASSIGNMENTS[self.idx_ass_assignment]
        except IndexError:
            # if a IndexError is raised, it means that we reached the end of the list of ass_assignments
            # we then stop the loop
            raise StopIteration

        # read hdf files
        train_df = self.train_df[ass_assignment]
        submission_df = self.submission_df[ass_assignment]

        # select dates
        train_df = train_df[:self.week_begin]
        submission_df = submission_df[self.week_begin:self.week_end]

        # split data
        dates_train = train_df.index.values
        dates_predict = submission_df.index.values
        X_train = train_df.drop(["CSPL_RECEIVED_CALLS"], axis=1)
        X_predict = submission_df.drop(["prediction"], axis=1)
        y_train = train_df["CSPL_RECEIVED_CALLS"].values

        return ass_assignment, dates_train, X_train, y_train, dates_predict, X_predict
