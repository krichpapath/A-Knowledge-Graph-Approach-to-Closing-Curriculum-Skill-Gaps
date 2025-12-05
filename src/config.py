# src/config.py

import os
from dotenv import load_dotenv

load_dotenv()

# ====== Data files ======
JOBS_FILE = os.getenv("JOBS_FILE")
COURSES_FILE = os.getenv("COURSES_FILE")

# ====== Embedding model ======
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# ====== Neo4j connection ======
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# 🔹 New: target database / graph name
NEO4J_GRAPH_NAME = os.getenv("NEO4J_GRAPH_NAME", "neo4j") 
