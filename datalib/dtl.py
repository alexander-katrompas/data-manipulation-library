"""
Author: Alex Katrompas

Data Types for use with dml
"""

from enum import Enum

class DataType(Enum):
    """
    Enumeration constants for identifying data types
    """
    NONE = 0
    NUMPY = 1
    DATAFRAME = 2
