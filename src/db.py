from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

class VectorDB:
    def __init__(self, persist_directory = "db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        os.makedirs(self.persist_directory, exist_ok = True )
        
        self.vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=self.embeddings, 
            persist_directory = self.persist_directory
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    def upload_document(self, path_to_single_document):
        loader = PyPDFLoader(path_to_single_document)
        documents = loader.load()
        docs = self.text_splitter.split_documents(documents)

        # Store docs into Chroma - persistence is automatic now
        self.vector_store.add_documents(docs)
        
    def upload_documents(self, documents_paths):
        for path in documents_paths:
            self.upload_document(path)
            
    def retrieve_context(self, query, k=3):
        docs = self.vector_store.similarity_search(query, k=k)
        context = "\n---\n".join([doc.page_content for doc in docs])
        
        # Extract source information from documents
        sources = []
        for doc in docs:
            source_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 1),
                "content": doc.page_content[:200] + "..."  # Preview of content
            }
            sources.append(source_info)
            
        return context, sources
