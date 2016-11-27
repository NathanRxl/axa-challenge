from preprocessing import Preprocesser
from time import time

initial_time = time()

preprocesser = Preprocesser()

# Clean csv step
print("Clean train csv...")
source_path = "data/train_2011_2012_2013.csv"
destination_path = "data/clean_train.csv"
sep = ";"
usecols = ["DATE", "ASS_ASSIGNMENT", "CSPL_RECEIVED_CALLS"]
preprocesser.clean_csv(source_path, destination_path, sep, usecols, group_by=True)

print("Clean sub csv...")
source_path = "data/submission.txt"
destination_path = "data/clean_submission.csv"
sep = "\t"
usecols = ["DATE", "ASS_ASSIGNMENT", "prediction"]
preprocesser.clean_csv(source_path, destination_path, sep, usecols)

# Create temporal features
print("\nCreate temporal features for train dataset...")
source_path = "data/clean_train.csv"
destination_path = "data/preprocessed_train.csv"
preprocesser.create_temporal_features(source_path, destination_path)

print("\nCreate temporal features for sub dataset...")
source_path = "data/clean_submission.csv"
destination_path = "data/preprocessed_submission.csv"
preprocesser.create_temporal_features(source_path, destination_path)

# Csv to hdf file step
print("\nConvert train csv file into hdf file...")
source_path = "data/preprocessed_train.csv"
destination_path = "data/train.h5"
preprocesser.csv_to_hdf(source_path, destination_path)

print("\nConvert sub csv file into hdf file...")
source_path = "data/preprocessed_submission.csv"
destination_path = "data/submission.h5"
preprocesser.csv_to_hdf(source_path, destination_path)

print("Preprocessing completed in %0.2f seconds" % (time() - initial_time))
