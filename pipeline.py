# call libraries:
import os
import joblib
from dotenv import load_dotenv

import pandas as pd
from pathlib import Path
from typing import Union

import sentence_transformers
from sentence_transformers import SentenceTransformer
import faiss
import openai

#output_dir = Path("D:/Data & Projects/medicortex.ai/model_objects")
#output_dir.mkdir(exist_ok=True)

# call modules & methods:
import dataprocessing
from dataprocessing import *

import index_embedding
from index_embedding import *

import build_FASSI_index
from build_FASSI_index import *

# load environment variables
load_dotenv()
output_dir = Path(os.getenv("OUTPUT_DIR"))
output_dir.mkdir(exist_ok=True)
input_load = os.getenv("INPUT_LOAD")

# pipeline:
#input_load = r"D:\Data & Projects\medicortex.ai\data\\rawdata.csv"
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

# Save embeddings
joblib.dump(desc_embeddings, output_dir / "desc_embeddings.pkl")
joblib.dump(patient_embeddings, output_dir / "patient_embeddings.pkl")
joblib.dump(doctor_embeddings, output_dir / "doctor_embeddings.pkl")


# Build FASSI index for 'Description' embeddings
desc_index = build_FASSI_index(desc_embeddings)
print(desc_index)

# Build FASSI index for 'Patient' embeddings
patient_index = build_FASSI_index(patient_embeddings)
print(patient_index)

# Build FASSI index for 'Doctor' embeddings
doctor_index = build_FASSI_index(doctor_embeddings)
print(doctor_index)

# Save indexes
faiss.write_index(desc_index, str(output_dir / "desc_index.faiss"))
faiss.write_index(patient_index, str(output_dir / "patient_index.faiss"))
faiss.write_index(doctor_index, str(output_dir / "doctor_index.faiss"))
# --- IGNORE ---