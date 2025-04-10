from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorDB:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=self.embeddings
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        
    def load_document(self, path_to_single_document):
        loader = PyPDFLoader(path_to_single_document)
        documents = loader.load()
        docs = self.text_splitter.split_documents(documents)

        # Store docs into Chroma
        self.vector_store.add_documents(docs)
        #vector_store.persist()

    def retrieve_context(self, query, k=3):
        docs = self.vector_store.similarity_search(query, k=k) # top 3 docs 
        context = "\n---\n".join([doc.page_content for doc in docs])
        return context 
