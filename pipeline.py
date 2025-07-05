python

#%%
# call libraries:
import pandas as pd
from pathlib import Path
from typing import Union


# call modules & methods:
import dataprocessing
from dataprocessing import *

# pipeline:
input_load = "D:\Data & Projects\medicortex.ai\data\\rawdata.csv"
data = load_and_process_data(input_load)