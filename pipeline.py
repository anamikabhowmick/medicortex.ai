


# call libraries:
import pandas as pd
from pathlib import Path
from typing import Union


# call modules & methods:
import dataprocessing
from dataprocessing import *
import index_embedding
from index_embedding import *

# pipeline:
input_load = "D:\Data & Projects\medicortex.ai\data\\rawdata.csv"
data = load_and_process_data(input_load)

# For 'Description' column:
desc_embeddings = sentence_embedding(df, 'Description')
print(desc_embeddings)

# For 'Patient' column:
patient_embeddings = sentence_embedding(df, 'Patient')
print(patient_embeddings)

# For 'Doctor' column:
doctor_embeddings = sentence_embedding(df, 'Doctor')
print(doctor_embeddings)