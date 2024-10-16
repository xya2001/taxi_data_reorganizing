import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


class NYCTaxiExampleDataset(torch.utils.data.Dataset):
    """Trainin data object for our nyc taxi data"""
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        # store the training data feature and label
        self.X_train = X_train
        self.y_train = y_train
        # define one hot encoder that transforms categorical data, and unknown categories are ignored 
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.X = torch.from_numpy(self._one_hot_X().toarray()) # potentially smarter ways to deal with sparse here
        self.y = torch.from_numpy(self.y_train.values)
        self.X_enc_shape = self.X.shape[-1] # 
        print(f"encoded shape is {self.X_enc_shape}") # Stores the shape of encoded X matrix
    
    def __len__(self):
        # get the number of samples
        return len(self.X)

    def __getitem__(self, i):
        # get the i th sample (both feature and label) from the dataset
        return self.X[i], self.y[i]
        
    def _one_hot_X(self):
        # encoding categorical features into one hot encoded binary features
        return self.one_hot_encoder.fit_transform(self.X_train)

class MLP(nn.Module):
    """Multilayer Perceptron for regression. """
    def __init__(self, encoded_shape):
        super().__init__()
        # Define the feed forward neural network process
        self.layers = nn.Sequential(
            nn.Linear(encoded_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1))
    
    def forward(self, x):
        return self.layers(x)
