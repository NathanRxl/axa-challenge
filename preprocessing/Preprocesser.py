import pandas as pd
import numpy as np
from workalendar.europe import France


class Preprocesser:
    """ Class performing preprocessing of the data for both train and submission files

    NB: during all preprocess pipeline, the dataframe is modified directly (not a copy of it) to prevent memory issues
    """

    def cast_dates_into_timestamp(self):
        # convert dates into pandas.Timestamp
        self.df["DATE"] = self.df["DATE"].apply(lambda date: pd.Timestamp(date))

    def drop_useless_columns(self):
        if "CSPL_RECEIVED_CALLS" in self.df.columns:
            self.df = self.df[["DATE", "ASS_ASSIGNMENT", "CSPL_RECEIVED_CALLS"]]
        else:
            self.df = self.df[["DATE", "ASS_ASSIGNMENT", "prediction"]]

    def split_dates(self):
        """ Create columns YEAR, MONTH, DAY, HOUR and MINUTE from DATE. """

        # split dates
        self.df["YEAR"] = self.df["DATE"].apply(lambda date: date.year)
        self.df["MONTH"] = self.df["DATE"].apply(lambda date: date.month)
        self.df["DAY"] = self.df["DATE"].apply(lambda date: date.day)
        self.df["HOUR"] = self.df["DATE"].apply(lambda date: date.hour)
        self.df["MINUTE"] = self.df["DATE"].apply(lambda date: date.minute)

    def __is_daytime(self, date):
        """ Function used inside fill_not_working_days method to convert a date into a boolean representing the daytime (1 for daytime, 0 for nighttime)"""
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
        self.df["DAY_OFF"] = self.df["DATE"].apply(lambda date: int(cal.is_holiday(date.to_datetime())))

        # week-end
        self.df["WEEK_END"] = self.df["DATE"].apply(lambda date: int(date.dayofweek in [5, 6]))

        # not working day
        self.df["NOT_WORKING_DAY"] = self.df["DAY_OFF"] | self.df["WEEK_END"]

        # daytime
        self.df["DAYTIME"] = self.df["DATE"].apply(lambda date: self.__is_daytime(date))

        # TODO
        # long week-ends

        # holidays

    def clean_csv(self, source_path, destination_path, sep, batch_size=100000, verbose=False):

        # read first data
        self.df = pd.read_csv(source_path, sep=sep, nrows=batch_size)

        # keep track of number of batchs done
        batch_nb = 0
        while True:
            if verbose: print("Batch #%d..." % batch_nb)

            # drop useless columns
            self.drop_useless_columns()

            # append processed data to csv
            with open(destination_path, "w" if batch_nb == 0 else "a") as file:
                header = (batch_nb == 0)
                self.df.to_csv(file, index=False, header=header)

            # if df has not the size of batch_size, it means we just preprocessed the last batch -> we end the loop
            if batch_size != len(self.df):
                break

            # update batch counter
            batch_nb += 1

            # read next batch data
            self.df = pd.read_csv(source_path, sep=sep, nrows=batch_size, skiprows=np.arange(1, batch_size * batch_nb))


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

    def group_by_date_ass_assignment(self, source_path, destination_path):
        # read csv
        self.df = pd.read_csv(source_path)

        # group by and sum received calls
        group_by = self.df.groupby(["DATE", "ASS_ASSIGNMENT"])
        self.df = pd.DataFrame(group_by["CSPL_RECEIVED_CALLS"].sum()).reset_index()

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
            hdf_file[ass_assignment] = self.df[self.df["ASS_ASSIGNMENT"] == ass_assignment].drop("ASS_ASSIGNMENT", axis=1)

        # close hdf file
        hdf_file.close()
