from preprocessing import Preprocesser

preprocesser = Preprocesser()

# train
source_path = "data/train_2011_2012_2013.csv"
destination_path = "data/preprocessed_train_2011_2012_2013.csv"
sep = ";"
batch_size = 500000

preprocesser.process_data(source_path, destination_path, sep, batch_size=batch_size, verbose=True)

# submission
source_path = "data/submission.txt"
destination_path = "data/preprocessed_submission.csv"
sep = "\t"
batch_size = 50000

preprocesser.process_data(source_path, destination_path, sep, batch_size=batch_size, verbose=True)
