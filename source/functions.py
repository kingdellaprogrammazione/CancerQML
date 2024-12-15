from __future__ import annotations  # PEP 563: Delayed type hint evaluation

# Third-Party Library Imports
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def data_split(file_name):

    df = pd.read_csv(file_name)

    # Split data into features (X) and target (y)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    return {
        'X_train': X_train, 
        'X_test': X_test, 
        'y_train': y_train, 
        'y_test': y_test, 
        'X_valid': X_valid, 
        'y_valid': y_valid
    }

def angle_encode(data):

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    X_valid = data['X_valid']
    y_valid = data['y_valid']
    
    # Initialize MinMaxScaler with a range of (-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Fit and transform the training data, then transform the test data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_valid = scaler.transform(X_valid)

    # Convert the arrays into PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid.values, dtype=torch.float32).view(-1, 1)
    # Return all six datasets
    return {
        'X_train': X_train, 
        'X_test': X_test, 
        'y_train': y_train, 
        'y_test': y_test, 
        'X_valid': X_valid, 
        'y_valid': y_valid
    }



def name_parser(file_path):
    file_name = file_path.name
    pieces = file_name.split('-')
    return pieces
