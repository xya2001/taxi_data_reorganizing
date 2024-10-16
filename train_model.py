import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder
import random
import numpy as np
from scripts.data_utils import raw_taxi_df, clean_taxi_df, split_taxi_data
from scripts.model_utils import NYCTaxiExampleDataset, MLP



def main(train_size, batch_size, max_epoch, learning_rate):
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # load data
    raw_df = raw_taxi_df(filename="./data/yellow_tripdata_2024-01.parquet")

    # data cleaning
    clean_df = clean_taxi_df(raw_df=raw_df)

    # split data
    location_ids = ['PULocationID', 'DOLocationID']
    # we want to find the relationship between fare amount and pick up, drop off locations.
    X_train, X_test, y_train, y_test = split_taxi_data(clean_df=clean_df, 
                                                    x_columns=location_ids, 
                                                    y_column="fare_amount", 
                                                    train_size=train_size)

    # Pytorch
    ## get data for train and test and decide the batch size
    dataset = NYCTaxiExampleDataset(X_train=X_train, y_train=y_train)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # Initialize the MLP
    mlp = MLP(encoded_shape=dataset.X_enc_shape)

    # Define the loss function and optimizer
    ## we want a loss function that is robust to the outliers
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

    # Run the training loop
    for epoch in range(0, max_epoch): # 5 epochs at maximum
        print(f'Starting epoch {epoch+1}')
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Perform forward pass
            outputs = mlp(inputs)
            
            # Compute loss
            loss = loss_function(outputs, targets)
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            
            # Print statistics
            current_loss += loss.item()
            if i % 10 == 0:
                print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500))
            current_loss = 0.0
    # Process is complete.
    print('Training process has finished.')
    torch.save(mlp, "./models/trained_model.pth")
    return X_train, X_test, y_train, y_test, data, mlp


main(train_size = 50000, batch_size = 10, max_epoch = 5, learning_rate = 1e-4)