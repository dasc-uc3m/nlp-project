import sys
sys.path.append(".")
from src.db import VectorDB

db = VectorDB()
db.upload_documents("/home/carlangas/Documents/Master/NLP/Documents/data/raw_pdfs")