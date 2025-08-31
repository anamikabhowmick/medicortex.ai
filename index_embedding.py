import pandas as pd
import sentence_transformers
from sentence_transformers import SentenceTransformer
import faiss
import openai

# Step 2: Load and preprocess data
df = pd.read_csv(r"D:\Data & Projects\medicortex.ai\data\\rawdata.csv")
df['text'] = df['Patient'].astype(str) + " " + df['Description'].astype(str)
print(df)

## testing embedding:
def sentence_embedding(df, column_name):
    """
    Computes sentence embeddings for the specified column in the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        column_name (str): Name of the column to embed (e.g., 'Description', 'Patient', 'Doctor').

    Returns:
        np.ndarray: Array of sentence embeddings.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Extract the relevant text column and convert to list
    sentences = df[column_name].tolist()
    # Generate embeddings
    embeddings = model.encode(sentences, show_progress_bar=True)
    return embeddings

# For 'Description' column:# For 'Patient' column:
desc_embeddings = sentence_embedding(df, 'Description')
print(desc_embeddings)

# For 'Patient' column:
patient_embeddings = sentence_embedding(df, 'Patient')
print(patient_embeddings)

# For 'Doctor' column:
doctor_embeddings = sentence_embedding(df, 'Doctor')
print(doctor_embeddings)
