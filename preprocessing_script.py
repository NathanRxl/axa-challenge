from preprocessing import Preprocessor
from time import time

initial_time = time()

preprocessor = Preprocessor()

# Preprocess train
source_path = "data/train_2011_2012_2013.csv"
destination_path = "data/train.h5"
sep = ";"
usecols = ["DATE", "ASS_ASSIGNMENT", "CSPL_RECEIVED_CALLS"]
preprocessor.preprocess_raw_data('train', source_path, sep, usecols, destination_path, group_by=True, debug=False)

# Preprocess submission
source_path = "data/submission.txt"
destination_path = "data/submission.h5"
sep = "\t"
usecols = ["DATE", "ASS_ASSIGNMENT", "prediction"]
preprocessor.preprocess_raw_data('submission', source_path, sep, usecols, destination_path, group_by=True, debug=False)

print("Preprocessing completed in %0.2f seconds" % (time() - initial_time))
