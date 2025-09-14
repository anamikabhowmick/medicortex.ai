# call libraries:
import pandas as pd
from pathlib import Path
from typing import Union

import sentence_transformers
from sentence_transformers import SentenceTransformer
import faiss
import openai



# call modules & methods:
import dataprocessing
from dataprocessing import *

import index_embedding
from index_embedding import *

# pipeline:
input_load = r"D:\Data & Projects\medicortex.ai\data\\rawdata.csv"
data = load_and_process_data(input_load)

# Run index embedding for all conversation attributes
# 'Description' column:
desc_embeddings = sentence_embedding(data, 'Description')
print(desc_embeddings)

# 'Patient' column:
patient_embeddings = sentence_embedding(data, 'Patient')
print(patient_embeddings)

# 'Doctor' column:
doctor_embeddings = sentence_embedding(data, 'Doctor')
print(doctor_embeddings)
