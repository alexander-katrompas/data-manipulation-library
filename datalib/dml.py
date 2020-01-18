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

DEFAULT_TRAIN_PCT = 80.0

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
    Parameters: Pandas DataFrame or Numby,
                optional np_array flag for return type, optional scale flag
    Processing: Will normalize and optionally scale the dataset
    Return: Pandas DataFrame or Numpy array with normalized/scaled data
    """
    # ensure floats
    dataset = dataset.astype(float)
    
    if detect_datatype(dataset) == DataType.NUMPY:
        # set-up normalization
        high = 1.0
        low = 0.0
        mins = np.min(dataset, axis=0)
        maxs = np.max(dataset, axis=0)
        rng = maxs - mins
        # normalize
        dataset = high - (((high - low) * (maxs - dataset)) / rng)
        # scale if needed
        if scaled:
            dataset = (dataset / .5) - 1
    elif detect_datatype(dataset) == DataType.DATAFRAME:
        # normalize
        [dataset[col].update((dataset[col] - dataset[col].min()) / (dataset[col].max() - dataset[col].min())) for col in dataset.columns]
        # scale if needed
        if scaled:
            [dataset[col].update((dataset[col] / .5)-1) for col in dataset.columns]
    
    # return appropriate object
    if np_array and detect_datatype(dataset) == DataType.DATAFRAME:
        dataset = dataset.to_numpy()
    elif not np_array and detect_datatype(dataset) == DataType.NUMPY:
        dataset = pd.DataFrame(dataset)
    
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
            X = data[:, :x_cols]
            Y = data[:, x_cols:]
        elif detect_datatype(data) == DataType.DATAFRAME:
            # left of the , ommitting start and stop gives "all rows"
            # right of the , ommitting start and including number of columns
            X = data.iloc[:, :x_cols]
            Y = data.iloc[:, x_cols:]

    return X, Y

def column_count(data):
    """
    Parameters: Pandas DataFrame or Numpy array
    Processing: Will count columns
    Return: Number of columns
    """
    column_count = 0
    
    if detect_datatype(data) == DataType.NUMPY:
        column_count = len(data[0])
    elif detect_datatype(data) == DataType.DATAFRAME:
        column_count = len(data.columns)

    return column_count

def split_dataset(data, train_pct=DEFAULT_TRAIN_PCT, random=False):
    """
    Parameters: Pandas DataFrame or Numpy array, percent of train data,
                random sampling flag to determin if taking random
                percent or first percent 
    Processing: Split data set
    Return: Training and test data sets of the same type input
    """
    train_data = None
    test_data = None

    # ensure train_pct is in range and convert to percent or default to .8
    try:
        train_pct = float(train_pct)
        if 0.0 < train_pct < 100.0:
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
            train_data, test_data = data[:train_rows, :], data[train_rows:, :]
        
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

def make_timeseries(data, out_cols=None, lag=1, fill=False):
    """
    Parameters: Pandas DataFrame or Numpy array, list of cols to extract,
                lag between I/O, fill data
    Processing: pull out output cols, shift them, append them back to the dataset
    Return: Dataset with ime shifted output
    """
    
    num_columns = column_count(data)
    
    # make sure lag is correct type and preserves at least half of data
    # else default to lag = 1
    if type(lag) != type(1) or (lag > int(data.shape[0] / 2) or lag < 1):
        lag = 1
    
    # make sure out_cols is valid
    if type(out_cols) == type(np.empty(0)) and out_cols.dtype == np.empty(0, dtype=int).dtype:
        # ensure unique list in order
        out_cols = np.sort(np.unique(out_cols))
        # remove any numbers not in range
        out_cols = out_cols[out_cols >= 0]
        out_cols = out_cols[out_cols < num_columns]
        # make sure len(out_cols) is smaller than num_columns
        out_cols = out_cols[:num_columns-1]
    else:
        # assume it's the last col if we don't have a valid list
        out_cols = [num_columns-1]

    if detect_datatype(data) == DataType.NUMPY:
        # extract output columns
        out_data = data[:, out_cols]
        
        #created data to insert for shift
        insert_data = np.empty((lag, column_count(data), ))
        insert_data[:] = np.NaN
        
        # add insert data to top
        data = np.insert(data, 0, insert_data, axis=0)
        
        #remove bottom of data
        data = data[:data.shape[0]-lag, :]
        
        # put them together
        data = np.concatenate((data, out_data), axis=1)
        
        if fill: data = np.nan_to_num(data)
        
    elif detect_datatype(data) == DataType.DATAFRAME:
        # extract output columns
        out_data = data.iloc[:, out_cols]

        #fix col name of out data
        out_data = out_data.add_suffix('_t-' + str(lag))

        #shift data
        data = data.shift(lag)

        # put them together
        data = pd.concat([data, out_data], axis=1)

        if fill: data.fillna(0, inplace=True)
    
    return data

def getColNames(data):
    col_names = []
    if detect_datatype(data) == DataType.NUMPY:
        col_names = list(map(str, list(range(0, column_count(data)))))
    elif detect_datatype(data) == DataType.DATAFRAME:
        col_names = list(map(str, data.columns))
    return col_names

def getMinMax(data):
    minmax = [[], []]
    if detect_datatype(data) == DataType.NUMPY:
        minmax[0] = np.amin(data, axis=0)
        minmax[1] = np.amax(data, axis=0)
    elif detect_datatype(data) == DataType.DATAFRAME:
        minmax = data.agg([min, max]).to_numpy()
    return minmax

def getStdev(data):
    stdev = []
    if detect_datatype(data) == DataType.NUMPY:
        stdev = np.nanstd(data, axis=0)
    elif detect_datatype(data) == DataType.DATAFRAME:
        #minmax = data.agg([min, max]).to_numpy()
        stdev = list(data.std(axis=0, skipna=True))
    return stdev

