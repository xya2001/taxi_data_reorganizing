Run train_model.py file to train the model for yellow trip data.

First we use read_parquet to load the data. Then the trips longer than 100 are removed from raw data, and columns for travel time deltas and time minutes are added. The data is then split to train and test data.

After cleaning the data, it is trained in batches using a multilayer perceptron class. 