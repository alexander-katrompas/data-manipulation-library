"""
Author: Alex Katrompas

Test script for dml and dtl
"""

import dml
import pandas as pd

DATAFILE = "../testdata/sample01a.csv"

dataset = pd.read_csv(DATAFILE, index_col=0)
print("Original Data")
print("==========================================")
print(dataset)
print()

dataset = dml.normalize(dataset, True)
print("Normalized Data")
print("==========================================")
print(dataset)
print()

train, test = dml.split_dataset(dataset, random=True)
print("Data Shape")
print("==========================================")
print(dataset.shape)
print(train.shape, test.shape)
print()

print("Data")
print("==========================================")
print(train)
print()
print(test)
print()

X_train, Y_train = dml.getio(train, 2)
X_test, Y_test = dml.getio(test, 2)
print("IO Data")
print("==========================================")
print(X_train)
print(Y_train)
print()
print(X_test)
print(Y_test)
print()
print("\n")
