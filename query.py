import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_answer(query):

    query_embedding = sentence_embedding([query])
    index_path = os.getenv("DOCTOR_INDEX_PATH")
    index = joblib.load(index_path)
    D, I = index.search(query_embedding, k=1)
    answer = df.iloc[I[0][0]]['Doctor Response']

    prompt = f"""You are a helpful medical assistant. The patient asked: "{query}". 
    Based on the following doctor's response, search the internet, 
    use chain of thoughts & reply professionally:\n\n{answer}"""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']
