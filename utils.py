import requests
import json
from config import LLM_BASE_URL, LLM_API_KEY, EMBEDDING_MODEL_NAME
from sentence_transformers import SentenceTransformer

# Load embedding model globally to avoid reloading
print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Embedding model loaded.")

def get_embedding(text):
    """Generates an embedding for the given text."""
    return embedding_model.encode(text).tolist()

def query_llm(prompt, system_message="You are a helpful assistant.", max_tokens=512, temperature=0.7):
    """
    Sends a request to the local LM Studio instance.
    """
    url = f"{LLM_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"
    }
    
    payload = {
        "model": "local-model", # Required by OpenAI-compatible APIs
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error querying LLM: {e}")
        if response is not None:
             print(f"Server Response: {response.text}")
        return None
