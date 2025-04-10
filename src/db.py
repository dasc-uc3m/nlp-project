from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings
)

loader = PyPDFLoader('data/hypertension.pdf')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Store docs into Chroma
vector_store.add_documents(docs)
#vector_store.persist()
