import pandas as pd
from workalendar.europe import France

# Dictionary with "new date feature column name" as keys and "attribute taken from date (pd.Timestamp)" as values
SPLIT_DATES = {
    "YEAR": "year",
    "MONTH": "month",
    "DAY": "day",
    "HOUR": "hour",
    "MINUTE": "minute",
    "DAY_OF_WEEK": "dayofweek",
}


class Preprocesser:
    """ Class performing preprocessing of the data for both train and submission files

    NB: during all preprocess pipeline, the dataframe is modified directly (not a copy of it) to prevent memory issues
    """

    def __init__(self):
        self.df = pd.DataFrame()

    def cast_dates_into_timestamp(self):
        # convert dates into pandas.Timestamp
        self.df["DATE"] = self.df["DATE"].apply(lambda date: pd.Timestamp(date))

    def split_dates(self):
        """ Create columns in SPLIT_DATES keys from DATE. """

        # split dates
        for new_date_feature in SPLIT_DATES:
            self.df[new_date_feature] = self.df["DATE"].apply(lambda date: getattr(date, SPLIT_DATES[new_date_feature]))

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
        self.df["DAY_OFF"] = self.df["DATE"].apply(lambda date: int(cal.is_holiday(date.to_pydatetime())))

        # week-end
        self.df["WEEK_END"] = self.df["DATE"].apply(lambda date: int(date.dayofweek in [5, 6]))

        # not working day
        self.df["NOT_WORKING_DAY"] = self.df["DAY_OFF"] | self.df["WEEK_END"]

        # daytime
        self.df["DAYTIME"] = self.df["DATE"].apply(lambda date: self.__is_daytime(date))

        # TODO
        # long week-ends

        # TODO
        # school holidays

    def clean_csv(self, source_path, destination_path, sep, usecols, group_by=False):
        """
        From raw csv dataset, create a clean csv by selecting useful columns and grouping by (DATE, ASS_ASSIGNMENT)
        """

        # read csv
        self.df = pd.read_csv(source_path, sep=sep, usecols=usecols)

        # group by and sum received calls
        if group_by:
            group_by = self.df.groupby(["DATE", "ASS_ASSIGNMENT"])
            self.df = pd.DataFrame(group_by["CSPL_RECEIVED_CALLS"].sum()).reset_index()

        # write csv
        self.df.to_csv(destination_path, index=False)

    def create_temporal_features(self, source_path, destination_path):
        # read csv
        self.df = pd.read_csv(source_path)

        # cast date into pandas.Timestamp
        self.cast_dates_into_timestamp()

        # date -> year, month,...
        self.split_dates()

        # fill day-off column
        self.fill_not_working_days()

        # write csv
        self.df.to_csv(destination_path, index=False)

    def csv_to_hdf(self, source_path, destination_path):
        # read csv
        self.df = pd.read_csv(source_path)

        # retrieve list of ass_assignments
        ass_assignments = self.df["ASS_ASSIGNMENT"].unique()

        # loop over ass_assignments to store into hdf_file
        hdf_file = pd.HDFStore(destination_path)
        for ass_assignment in ass_assignments:
            hdf_file[ass_assignment] = (
                self.df[self.df["ASS_ASSIGNMENT"] == ass_assignment].drop("ASS_ASSIGNMENT", axis=1)
            )

        # close hdf file
        hdf_file.close()
