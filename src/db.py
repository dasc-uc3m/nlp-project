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
        loader = PyPDFLoader(path_to_single_document, mode="single")
        try:
            documents = loader.load()
        except Exception as e:
            print(f"Document: {path_to_single_document} couldn't be generated.\nError: {e}.\n\n")
            return None
        docs = self.text_splitter.split_documents(documents)

        for idx, doc in enumerate(docs):
            # We add the chunk id as part of the metadata to save which part of the document this chunk is.
            doc.metadata["chunk_idx"] = idx
        # Store docs into Chroma - persistence is automatic now
        if len(docs) > 0:
            self.vector_store.add_documents(docs)
        
    def upload_documents(self, documents_paths):
        for path in os.listdir(documents_paths):
            self.upload_document(os.path.join(documents_paths, path))
            
    def retrieve_context(self, query, k=3, chunk_window_size=4):
        docs = self.vector_store.similarity_search(query, k=k)
        joined_chunks = []
        for doc in docs:
            joined_chunks.append(self._search_nearby_chunks(doc, chunk_window_size))
        context = "\n\n---\n\n".join(joined_chunks)
        
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

    def _search_nearby_chunks(self, doc, window):
        source_doc = doc.metadata["source"]
        chunk_idx = doc.metadata["chunk_idx"]
        # Get every chunk of the same pdf document.
        response = self.vector_store.get(where={"source": source_doc})

        all_chunks = zip(response["documents"], response["metadatas"])
        # Sort chunks by chunk_idx
        all_chunks = sorted(all_chunks, key=lambda c: float(c[1]["chunk_idx"]))

        nearby_chunks = []
        target_indices = set(range(chunk_idx - window, chunk_idx + window + 1))
        for doc in all_chunks:
            doc_meta = doc[1]
            if doc_meta["chunk_idx"] in target_indices:
                nearby_chunks.append(doc[0])

        return "\n".join(nearby_chunks)