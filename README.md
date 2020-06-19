# Data Manipulation Library
For Numpy arrays and Pandas DataFrames

dml.py
-----------------------------------------------------------------------
Library of functions to wrap common Numpy and Pandas operations in a single interface. The purpose being to be able to easily move from Numpy and Pandas without thinking about it, and to perform common matrix manipulations, and a few more interesting functions like make_timeseries() and normalize() (which will also scale).

prosproc.py
-----------------------------------------------------------------------
A class for post processing binary classification data producing all common statistics including confusion matrix, MSE (average and per sequence), ROC and PR curves and area-under-curves. Will categorize data into binary sequences for further analysis.

