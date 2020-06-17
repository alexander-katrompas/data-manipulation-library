"""
Author: Alex Katrompas

Post Processing Library
For time series binary classification data.
i.e. temporal classifaction
"""

import numpy as np

DEFAULT_THRESHHOLD = 0.7

def load_actual_predicted(filename):
    f = open(filename, "r")
    f.close()
