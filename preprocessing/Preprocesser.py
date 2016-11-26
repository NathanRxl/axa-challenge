import pandas as pd
import numpy as np
from workalendar.europe import France


class Preprocesser:
    """ Class performing preprocessing of the data for both train and submission files

    NB: during all preprocess pipeline, the dataframe is modified directly (not a copy of it) to prevent memory issues
    """

    def drop_useless_features(self):
        if "CSPL_RECEIVED_CALLS" in self.df.columns:
            self.df = self.df[["DATE", "ASS_ASSIGNMENT", "CSPL_RECEIVED_CALLS"]]
        else:
            self.df = self.df[["DATE", "ASS_ASSIGNMENT", "prediction"]]

    def split_dates(self):
        """ Create columns YEAR, MONTH, DAY, HOUR and MINUTE from DATE. """

        # convert dates into pandas.Timestamp
        self.df["DATE"] = self.df["DATE"].apply(lambda date: pd.Timestamp(date))

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

    def process_data(self, source_path, destination_path, sep, batch_size=100000, verbose=False):

        if verbose: print("Start preprocessing...")

        # read first data
        self.df = pd.read_csv(source_path, sep=sep, nrows=batch_size)

        # keep track of number of batchs done
        batch_nb = 0
        while True:

            if verbose: print("Batch #%d..." % batch_nb)

            # drop columns
            self.drop_useless_features()

            # date -> year, month,...
            self.split_dates()

            # fill day-off column
            self.fill_not_working_days()

            # append processed data to csv
            with open(destination_path, "w" if batch_nb == 0 else "a") as file:
                header = (batch_nb == 0)
                self.df.to_csv(file, index=False, header=header)

            # if df has not the size of batch_size, it we just preprocessed the last batch -> we end the loop
            if batch_size != len(self.df):
                break

            # update batch counter
            batch_nb += 1

            # read next batch data
            self.df = pd.read_csv(source_path, sep=sep, nrows=batch_size, skiprows=np.arange(1, batch_size * batch_nb))

        if verbose: print("Preprocessing done...")
