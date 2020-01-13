"""
Author: Alex Katrompas

Data Manipulation Library
For numpy arrays and pandas dataframes.
For use with machine learning and data science applications.
"""

from dtl import DataType
import numpy as np
import pandas as pd
np.set_printoptions(precision=4, floatmode="fixed")

DEFAULT_TRAIN_PCT = 80

def detect_datatype(data):
    """
    Parameters: Pandas DataFrame or Numpy array
    Processing: Detect type of Numpy or DataFrame type
    Return: DataType constant
    """
    if type(data) == type(np.empty(0)):
        return DataType.NUMPY
    elif type(data) == type(pd.DataFrame({'A': []})):
        return DataType.DATAFRAME

def normalize(dataset, np_array=False, scaled=False):
    """
    Parameters: Pandas DataFrame, optional np_array for return type, optional scale flag
    Processing: Will normalize and optionally scale a dataset
    Return: Pandas DataFrame or Numpy array with normalized/scaled data
    """
    # ensure floats
    dataset = dataset.astype(float)
    
    # normalize
    [dataset[col].update((dataset[col] - dataset[col].min()) / (dataset[col].max() - dataset[col].min())) for col in dataset.columns]
    
    # scale if needed
    if scaled:
        [dataset[col].update((dataset[col] / .5)-1) for col in dataset.columns]
    
    # return appropriate object
    if np_array:
        return dataset.to_numpy()
    else:
        return dataset

def getio(data, x_cols):
    """
    Parameters: Pandas DataFrame or Numpy array, number of column from left to right
           that are the input columns.
    Processing: Will slice into two sets, input and output data
    Return: Two data sets of the same type sent in, input and output
    """
    X = None
    Y = None
    total_cols = column_count(data)
    
    if (x_cols > 0 and x_cols < total_cols):
        if detect_datatype(data) == DataType.NUMPY:
            X = data[:,:x_cols]
            Y = data[:, x_cols:]
        elif detect_datatype(data) == DataType.DATAFRAME:
            # left of the , ommitting start and stop gives "all rows"
            # right of the , ommitting start and including number of columns
            X = data.iloc[:,:x_cols]
            Y = data.iloc[:, x_cols:]

    return X, Y

def column_count(data):
    """
    Parameters: Pandas DataFrame or Numpy array
    Processing: Will count columns
    Return: Number of columns
    """
    if detect_datatype(data) == DataType.NUMPY:
        return len(data[0])
    elif detect_datatype(data) == DataType.DATAFRAME:
        return len(data.columns)
    else:
        return 0
    
def split_dataset(data, train_pct=DEFAULT_TRAIN_PCT, random=False):
    """
    Parameters: Pandas DataFrame or Numpy array, percent of train data,
                random sampling flag to determin if taking random
                percent or first percent 
    Processing: Split data set
    Return: trianing and test data sets of the same type input
    """
    train_data = None
    test_data = None

    # ensure train_pct is in range and convert to percent or default to .8
    try:
        train_pct = int(train_pct)
        if 0 < train_pct < 100:
            train_pct = float(train_pct) / 100.0
        else:
            train_pct = float(DEFAULT_TRAIN_PCT) / 100.0
    except:
        train_pct = float(DEFAULT_TRAIN_PCT) / 100.0
    
    # detect data type and split based on data type
    if detect_datatype(data) == DataType.NUMPY:
        if random:
            # this violates PEP 8 and slows the function,
            # but it is called rarely and I like it here better.
            from sklearn.model_selection import train_test_split
            train_data, test_data = train_test_split(data, train_size=train_pct)
        else:
            # split numpy array
            rows = data.shape[0]
            train_rows = int(rows * train_pct)
            train_data, test_data = data[:train_rows,:], data[train_rows:,:]
        
    elif detect_datatype(data) == DataType.DATAFRAME:
        # this violates PEP 8 and slows the function,
        # but it is called rarely and I like it here better.
        from sklearn.model_selection import train_test_split
        # split DataFrame
        if random:
            train_data, test_data = train_test_split(data, train_size=train_pct)
        else:
            train_data, test_data = train_test_split(data, train_size=train_pct, shuffle=False)

    return train_data, test_data