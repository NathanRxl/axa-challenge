import numpy as np
import pandas as pd
from workalendar.europe import France
import warnings


class Preprocessor:
    """ Class performing preprocessing of the data for both train and submission files

    NB: during all preprocess pipeline, the dataframe is modified directly (not a copy of it) to prevent memory issues
    """

    # Dictionary with "new date feature column name" as keys and "attribute taken from date (pd.Timestamp)" as values
    SPLIT_DATES = {
        "YEAR": "year",
        "MONTH": "month",
        "DAY": "day",
        "HOUR": "hour",
        "MINUTE": "minute",
        "DAY_OF_WEEK": "dayofweek",
    }

    CATEGORICAL_FEATURES = [
        ("YEAR", pd.Series([2011, 2012, 2013])),
        ("MONTH", pd.Series(np.arange(12))),
        ("DAY", pd.Series(np.arange(31))),
        ("HOUR", pd.Series(np.arange(24))),
        ("MINUTE", pd.Series([0, 30])),
        ("DAY_OF_WEEK", pd.Series(np.arange(7))),
    ]

    def __init__(self):
        self.df = pd.DataFrame()

    def split_dates(self):
        """ Create columns in SPLIT_DATES keys from DATE. """
        # split dates
        for new_date_feature in self.SPLIT_DATES:
            self.df[new_date_feature] = getattr(self.df.index, self.SPLIT_DATES[new_date_feature])

    @staticmethod
    def __is_daytime(date):
        """
        Function used inside fill_not_working_days method to convert a date into a boolean representing the daytime
        (1 for daytime, 0 for nighttime)
        """
        if date.hour in [23, 0, 1, 2, 3, 4, 5, 6, 7]:
            if date.hour == 23 and date.minute == 0:
                return 1
            else:
                return 0
        else:
            return 1

    def fill_not_working_days(self):
        """ Create columns WEEK_END and DAY_OFF from DATE """
        # days-off
        cal = France()
        self.df["DAY_OFF"] = np.vectorize(cal.is_holiday)(self.df.index.to_pydatetime()).astype(int)
        # week-end
        self.df["WEEK_END"] = np.in1d(self.df.index.dayofweek, [5, 6]).astype(int)
        # not working day
        self.df["NOT_WORKING_DAY"] = self.df["DAY_OFF"] | self.df["WEEK_END"]
        # daytime
        self.df["DAYTIME"] = self.df.index.map(self.__is_daytime)
        # TODO
        # long week-ends
        # TODO
        # school holidays

    def create_dummy(self, drop_first=True):
        from time import time
        """ Create dummy features from categorical features """
        for feature_name, feature_values in self.CATEGORICAL_FEATURES:
            nb_possible_values = len(feature_values)
            # append every possible values of the feature to real feature column
            enhanced_feature_series = self.df[feature_name].append(feature_values)
            # get dummy features
            dummy_features_df = pd.get_dummies(enhanced_feature_series, prefix=feature_name, drop_first=drop_first)[:-nb_possible_values]
            # drop old feature column and add dummy features
            self.df.drop(feature_name, axis=1, inplace=True)
            self.df[dummy_features_df.columns] = dummy_features_df.astype(int)

    def preprocess_raw_data(self, train_or_submission, source_path, sep, usecols, destination_path,
                            group_by=False, dummy_features=False, debug=False):
        # Clean csv
        print("Clean {} csv...".format(train_or_submission))
        # read csv
        self.df = pd.read_csv(source_path, sep=sep, parse_dates=True, index_col="DATE", usecols=usecols)
        # group by and sum received calls
        if group_by:
            self.df = (
                self.df.groupby(by=[self.df.index, "ASS_ASSIGNMENT"])
                    .sum().reset_index(level="ASS_ASSIGNMENT")
            )
        if debug:
            self.df.to_csv("data/clean_{}.csv".format(train_or_submission), index=True)

        # Create temporal features
        print("Create temporal features for {} dataset...".format(train_or_submission))
        # date -> year, month,...
        self.split_dates()
        # fill day-off column
        self.fill_not_working_days()
        # create dummy features
        if dummy_features:
            self.create_dummy()
        if debug:
            # write csv
            self.df.to_csv("data/preprocessed_{}.csv".format(train_or_submission), index=True)

        # Convert csv file into hdf file
        print("Convert {} csv file into hdf file...".format(train_or_submission))
        # retrieve list of ass_assignments
        ass_assignments = self.df["ASS_ASSIGNMENT"].unique()
        # loop over ass_assignments to store into hdf_file
        warnings.filterwarnings('ignore', 'object name is not a valid Python identifier')
        hdf_file = pd.HDFStore(destination_path)
        for ass_assignment in ass_assignments:
            hdf_file[ass_assignment] = (
                self.df[self.df["ASS_ASSIGNMENT"] == ass_assignment].drop("ASS_ASSIGNMENT", axis=1)
            )
        # close hdf file
        hdf_file.close()
        print()
