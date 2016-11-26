from preprocessing import Preprocesser

preprocesser = Preprocesser()

# Clean csv step

# train
source_path = "data/train_2011_2012_2013.csv"
destination_path = "data/clean_train.csv"
sep = ";"
batch_size = 500000

print("Clean train csv...")
preprocesser.clean_csv(source_path, destination_path, sep, batch_size=batch_size, verbose=True)

# submission
source_path = "data/submission.txt"
destination_path = "data/clean_submission.csv"
sep = "\t"
batch_size = 50000

print("\nClean sub csv...")
preprocesser.clean_csv(source_path, destination_path, sep, batch_size=batch_size, verbose=True)

# Group by tuple (DATE, ASS_ASSIGNMENT) for train dataset
print("\nGroup by (DATE, ASS_ASSIGNMENT) for train dataset...")
source_path = "data/clean_train.csv"
destination_path = "data/grouped_by_train.csv"
preprocesser.group_by_date_ass_assignment(source_path, destination_path)

# Create temporal features
print("\nCreate temporal features for train dataset...")
source_path = "data/grouped_by_train.csv"
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

