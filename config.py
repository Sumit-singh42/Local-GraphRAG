import os

# LM Studio Configuration
LLM_BASE_URL = "http://192.168.56.1:1234/v1"  # Updated IP from user
LLM_API_KEY = "lm-studio" # Usually ignored by local runners, but good protocol

# Embedding Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Fast, lightweight, good enough for MVP

# Vector Store Configuration
VECTOR_DB_CTX = "./chroma_db"

# Graph Configuration
GRAPH_STORAGE_PATH = "knowledge_graph.gml"
