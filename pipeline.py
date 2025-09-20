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
from tqdm import tqdm


# call modules & methods:
import dataprocessing
from dataprocessing import *

import index_embedding
from index_embedding import *

import build_FASSI_index
from build_FASSI_index import *

# load environment variables

load_dotenv()
output_dir_env = os.getenv("OUTPUT_DIR")
if output_dir_env is None:
	raise ValueError("Environment variable OUTPUT_DIR is not set. Please set it in your environment or .env file.")
output_dir = Path(output_dir_env)
output_dir.mkdir(exist_ok=True)
input_load = os.getenv("INPUT_LOAD")

# pipeline: load and process data
data = load_and_process_data(input_load)

# Run index embedding for all conversation attributes
# 'Description' column:
desc_embeddings = sentence_embedding(data, 'Description')
joblib.dump(desc_embeddings, output_dir / "desc_embeddings.pkl")
print(desc_embeddings)

# 'Patient' column:
patient_embeddings = sentence_embedding(data, 'Patient')
joblib.dump(patient_embeddings, output_dir / "patient_embeddings.pkl")
print(patient_embeddings)

# 'Doctor' column:
doctor_embeddings = sentence_embedding(data, 'Doctor')
joblib.dump(doctor_embeddings, output_dir / "doctor_embeddings.pkl")
print(doctor_embeddings)


# Build FASSI index for 'Description' embeddings
desc_index = build_FASSI_index(desc_embeddings)
faiss.write_index(desc_index, str(output_dir / "desc_index.faiss"))
print(desc_index)

# Build FASSI index for 'Patient' embeddings
patient_index = build_FASSI_index(patient_embeddings)
faiss.write_index(patient_index, str(output_dir / "patient_index.faiss"))
print(patient_index)

# Build FASSI index for 'Doctor' embeddings
doctor_index = build_FASSI_index(doctor_embeddings)
faiss.write_index(doctor_index, str(output_dir / "doctor_index.faiss"))
print(doctor_index)


# --- IGNORE ---