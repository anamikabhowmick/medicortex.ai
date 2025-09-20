import numpy as np
import joblib
import faiss
from pathlib import Path

# Load embeddings from model_output folder
model_output = Path('model_output')

desc_embeddings = joblib.load(model_output / 'desc_embeddings.pkl')
patient_embeddings = joblib.load(model_output / 'patient_embeddings.pkl')
doctor_embeddings = joblib.load(model_output / 'doctor_embeddings.pkl')

def test_embeddings(embeddings, name):
    print(f"{name} shape: {getattr(embeddings, 'shape', None)} | dtype: {getattr(embeddings, 'dtype', None)} | type: {type(embeddings)}")
    if embeddings.ndim != 2:
        print(f"ERROR: {name} is not 2D!")
    if embeddings.dtype != np.float32:
        print(f"ERROR: {name} is not float32!")
    else:
        print(f"{name} is valid.")
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        print(f"{name} FAISS index built. ntotal: {index.ntotal}")
    except Exception as e:
        print(f"Failed to build FAISS index for {name}: {e}")

for arr, name in [
    (desc_embeddings, 'desc_embeddings'),
    (patient_embeddings, 'patient_embeddings'),
    (doctor_embeddings, 'doctor_embeddings'),
]:
    test_embeddings(arr, name)
