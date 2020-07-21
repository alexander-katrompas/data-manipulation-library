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

# #############################################
# MANIPULATION FUNCTIONS
# These functions restructure the data
# in some way and return the manipulated data
# #############################################

def remove_bad_data(infile, outfile, labels=False, header=False):
    """
    Remove rows with missing or invalid numeric values.
    
    Parameters: input file name, output file name
                optional flags to indicate the presence of headers and labels
    Processing: Will remove rows that contain a non-numeric value and place the
                processed data into outfile.
    Return: a count of the total rows, and the count or rows removed
    """
    fin = open(infile)
    count_total = 0
    count_bad = 0

    if fin:
        fout = open(outfile, "w")
        if header: fin.readine() # throw away header
        for line in fin:
            line = line[:-1] # remove \n
            line = line.split(",")
            if labels:
                label = line[0] # save label
                line = line[1:] # throw away labels
            else:
                label = "" # dummy label
            
            length = len(line)
            good_line = True
            
            for i in range(length):
                try:
                    float(line[i])
                except:
                    good_line = False
            
            if not good_line:
                count_bad += 1
            else:
                fout.write(label + "," + ",".join(line) + "\n")
            count_total += 1
                
    return count_total, count_bad


def normalize(data, np_array=False, scaled=False):
    """
    Normalize and optionally scale a dataset.
    
    Parameters: Pandas DataFrame or Numpy,
                optional np_array flag for return type, optional scale flag
    Processing: Will normalize and optionally scale the dataset
    Return: Pandas DataFrame or Numpy array with normalized/scaled data
    """
    # ensure floats
    data = data.astype(float)
    
    if detect_datatype(data) == DataType.NUMPY:
        # set-up normalization
        high = 1.0
        low = 0.0
        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)
        rng = maxs - mins
        # normalize
        data = high - (((high - low) * (maxs - data)) / rng)
        # scale if needed
        if scaled:
            data = (data / .5) - 1
    elif detect_datatype(data) == DataType.DATAFRAME:
        # normalize
        [data[col].update((data[col] - data[col].min()) / (data[col].max() - data[col].min())) for col in data.columns]
        # scale if needed
        if scaled:
            [data[col].update((data[col] / .5)-1) for col in data.columns]
    
    # return appropriate object
    if np_array and detect_datatype(data) == DataType.DATAFRAME:
        data = data.to_numpy()
    elif not np_array and detect_datatype(data) == DataType.NUMPY:
        data = pd.DataFrame(data)
    
    return data

def getio(data, x_cols):
    """
    Slice a dataset into input and output features based on the last x_cols
    
    Parameters: Pandas DataFrame or Numpy array, number of column from left to right
           that are the input columns.
    Processing: Will slice into two sets, input and output data
    Return: Two data sets of the same type sent in, input and output
    """
    total_cols = column_count(data)
    if len(data.shape) != 2:
        raise TypeError("Input data must be 2D.")
    elif x_cols < 1 or x_cols >= total_cols:
        raise ValueError("Input column count must be between 1 and " + str(total_cols - 1) + " inclusive")

    if detect_datatype(data) == DataType.NUMPY:
        X = data[:,:x_cols]
        Y = data[:, x_cols:]
    elif detect_datatype(data) == DataType.DATAFRAME:
        # left of the , ommitting start and stop gives "all rows"
        # right of the , ommitting start and including number of columns
        X = data.iloc[:,:x_cols]
        Y = data.iloc[:, x_cols:]

    return X, Y

def split_dataset(data, train_pct=DEFAULT_TRAIN_PCT, random=False):
    """
    Split a dataset into test and training, either randomly or sequentially
    
    Parameters: Pandas DataFrame or Numpy array, percent of train data,
                random sampling flag to determin if taking random
                percent or first percent 
    Processing: Split data set into two data sets either randomly or sequentially
    Return: Training and test data sets of the same type input
    """
    if len(data.shape) != 2: raise TypeError("Input data must be 2D.")
    
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

def make_integer_data(data, cols, scale=10000):
    """
    Transform float data into integer data.
    
    Parameters: data: Pandas DataFrame or Numpy array
                cols: columns to scale (but all come back as ints)
                scale: factor to scale
    Processing: Make data into scaled integer data.
                For example: if scale is 1000 and data
                is .9 this will become 900
    Return: Dataset with scaled integers
    """
    if len(data.shape) != 2: raise TypeError("Input data must be 2D.")
    
    ncols = column_count(data)
    if cols > ncols: cols = ncols # protect the indexes
    
    if detect_datatype(data) == DataType.NUMPY:
        data[:, 0:cols] *= float(scale)
    elif detect_datatype(data) == DataType.DATAFRAME:
        # this works but causes a copy of a slice warning
        #column_names = getColNames(data)
        #column_names = column_names[:cols]
        #data.loc[:, column_names[0]:column_names[-1]] *= scale
        # this works too but causes a copy of a slice warning
        #data.loc[:, column_names[0]:column_names[-1]] = data.loc[:, column_names[0]:column_names[-1]] * scale
        
        # this also works but also causes a copy of a slice warning
        #data.iloc[:, 0:cols] = data.iloc[:, 0:cols].mul(float(scale))
        
        # this isn't very python like, but causes no warnings, using it until
        # I figure out how to do it the "Python way" without warnings
        rows = len(data.index)
        for row in range(0, rows):
            for col in range (0, cols):
                data.iat[row, col] = data.iat[row, col] * scale
    
    return data.astype(int)
    

def make_timeseries(data, out_cols=None, lag=1, fill=False):
    """
    Create timeseries data for prediction with the output being a time shifted version of an input column.
    
    Parameters: Pandas DataFrame or Numpy array, list of cols to extract,
                lag between I/O, fill data
    Processing: pull out output cols, shift them, append them back to the dataset
    Return: Dataset with time shifted output
    """
    if len(data.shape) != 2: raise TypeError("Input data must be 2D.")
    
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
        insert_data = np.empty((lag, column_count(data),))
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


def shape_3D_data(data, timesteps):
    if len(data.shape) != 2: raise TypeError("Input data must be 2D.")
    
    """
    Resape 2D data into 3D data of groups of 2D timesteps
    
    Parameters: Pandas DataFrame or Numpy 3D array, number of timesteps/group
    Processing: Reshape 2D data into 3D data of array of 2D 
    Return: The reshaped data as numpy 3D array
    """
    # time steps are steps per batch
    features = len(data[0])
    # samples are total number of input vectors
    samples = data.shape[0]
    
    # samples must divide evenly by timesteps to create an even set of batches
    if not(samples % timesteps):
        return np.array(data).reshape(int(data.shape[0] / timesteps), timesteps, features)
    else:
        msg = "timesteps must divide evenly into total samples: " + str(samples) + "/" \
            + str(timesteps) + "=" + str(round(float(samples) / float(timesteps), 2))
        raise ValueError(msg)


# #############################################
# INFORMATIONAL FUNCTIONS 
# These functions report on data and metadata
# #############################################

def count_unique(data):
    """
    Count the number of unique values in a Pandas DataFrame or Numpy array
    
    Parameters: Pandas DataFrame or Numpy array
    Processing: count unique values
    Return: count
    """

    count = 0
    if detect_datatype(data) == DataType.NUMPY:
        count = len(np.unique(data))
    elif detect_datatype(data) == DataType.DATAFRAME:
        count = count_unique(data.to_numpy())
    return count

def detect_datatype(data):
    """
    Find out if data is Pandas DataFrame or Numpy array
    
    Parameters: Pandas DataFrame or Numpy array
    Processing: Detect type of Numpy or DataFrame type
    Return: DataType constant
    """
    if type(data) == type(np.empty(0)):
        return DataType.NUMPY
    elif type(data) == type(pd.DataFrame({'A': []})):
        return DataType.DATAFRAME

def column_count(data):
    """
    Count the columns in a Pandas DataFrame or Numpy array
    
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

def get_col_names(data):
    """
    Extract the list of column names or ordinal numbers of columns
    
    Parameters: DataFrame or Numpy array
    Processing: Extract the list of column names
    Return: A list of column names
    """

    col_names = []
    if detect_datatype(data) == DataType.NUMPY:
        # "labels" columns 0,1,2, etc
        col_names = list(map(str, list(range(0, column_count(data)))))
    elif detect_datatype(data) == DataType.DATAFRAME:
        col_names = list(map(str, data.columns))
    return col_names

def get_min_max(data):
    """
    Find and return the min and max of a DataFrame or Numpy array
    
    Parameters: DataFrame or Numpy array
    Processing: Find min and max
    Return: min and max
    """

    minmax = [[], []]
    if detect_datatype(data) == DataType.NUMPY:
        minmax[0] = np.amin(data, axis=0)
        minmax[1] = np.amax(data, axis=0)
    elif detect_datatype(data) == DataType.DATAFRAME:
        minmax = data.agg([min, max]).to_numpy()
    return minmax

def get_stdev(data):
    """
    Calcuate the standard deviation of data
    
    Parameters: Numpy array or DataFrame
    Processing: Calcuate the standard deviation of data
    Return: the standard deviation
    """

    stdev = []
    if detect_datatype(data) == DataType.NUMPY:
        stdev = np.nanstd(data, axis=0)
    elif detect_datatype(data) == DataType.DATAFRAME:
        #minmax = data.agg([min, max]).to_numpy()
        stdev = list(data.std(axis=0, skipna=True))
    return stdev
