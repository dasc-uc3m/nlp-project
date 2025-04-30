import sys
import os
sys.path.append(".")
from src.db import VectorDB

def main():
    # Initialize the vector database
    db = VectorDB()
    
    # Get all documents
    print("\nFetching documents from the database...\n")
    
    # Get all vectors and their metadata
    results = db.vector_store.get()
    
    # Print document information
    if not results['ids']:
        print("No documents found in the database.")
        return
        
    print(f"Total number of chunks: {len(results['ids'])}")
    print("\nDocuments in database:")
    print("-" * 50)
    
    # Group chunks by source document
    documents = {}
    for i, source in enumerate(results['metadatas']):
        doc_source = source.get('source', 'Unknown')
        if doc_source not in documents:
            documents[doc_source] = {
                'chunks': 0,
                'pages': set()
            }
        documents[doc_source]['chunks'] += 1
        documents[doc_source]['pages'].add(source.get('page', 0))
    
    # Print document information
    for doc_source, info in documents.items():
        print(f"Document: {os.path.basename(doc_source)}")
        print(f"  - Number of chunks: {info['chunks']}")
        print(f"  - Pages: {sorted(info['pages'])}")
        print("-" * 50)

if __name__ == "__main__":
    main() 