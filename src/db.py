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
            # Extract just the filename from the full path
            full_source = doc.metadata.get("source", "Unknown")
            source_filename = os.path.basename(full_source)
            
            source_info = {
                "source": source_filename,
                "content": doc.page_content[:500] + "..."  # Preview of content
            }
            sources.append(source_info)
            
        return context, sources

    def _search_nearby_chunks(self, doc, window):
        source_doc = doc.metadata["source"]
        # Get every chunk of the same pdf document.
        all_chunks = self.vector_store.get(where={"source": source_doc})

        # Create a list of tuples (chunk_idx, content) for sorting
        chunk_data = []
        for i, content in enumerate(all_chunks["documents"]):
            metadata = all_chunks["metadatas"][i]
            # Use chunk_idx if available, otherwise use position in list
            chunk_idx = float(metadata.get("chunk_idx", i))
            chunk_data.append((chunk_idx, content))

        # Sort chunks by chunk_idx
        chunk_data.sort(key=lambda x: x[0])

        # Get the current chunk's index
        current_chunk_idx = float(doc.metadata.get("chunk_idx", 0))
        
        nearby_chunks = []
        target_indices = set(range(int(current_chunk_idx - window), int(current_chunk_idx + window + 1)))
        for idx, content in chunk_data:
            if idx in target_indices:
                nearby_chunks.append(content)

        return "\n".join(nearby_chunks)

    def list_documents(self):
        """List all documents in the vector store"""
        try:
            # Get all documents and their metadata
            results = self.vector_store.get()
            # Extract unique source documents
            documents = set()
            for metadata in results['metadatas']:
                if 'source' in metadata:
                    # Get just the filename from the full path
                    filename = os.path.basename(metadata['source'])
                    documents.add(filename)
            return list(documents)
        except Exception as e:
            print(f"Error listing documents: {str(e)}")
            return []

    def delete_document(self, filename):
        """Delete a document and all its chunks from the vector store"""
        try:
            print(f"DEBUG - Starting deletion for filename: {filename}")
            # Get all documents and their metadata
            results = self.vector_store.get()
            print(f"DEBUG - Retrieved {len(results['metadatas'])} total documents")
            
            # Find all chunk IDs that belong to this document
            ids_to_delete = []
            for i, metadata in enumerate(results['metadatas']):
                if 'source' in metadata:
                    source_filename = os.path.basename(metadata['source'])
                    print(f"DEBUG - Checking source: {source_filename} against target: {filename}")
                    if source_filename == filename:
                        ids_to_delete.append(results['ids'][i])
            
            print(f"DEBUG - Found {len(ids_to_delete)} chunks to delete")
            if not ids_to_delete:
                print(f"No document found with filename: {filename}")
                return False
            
            # Delete the chunks
            try:
                self.vector_store.delete(ids=ids_to_delete)
                print(f"Successfully deleted document: {filename}")
                return True
            except Exception as e:
                print(f"Error during deletion of document {filename}: {str(e)}")
                return False
            
        except Exception as e:
            print(f"Error deleting document {filename}: {str(e)}")
            return False