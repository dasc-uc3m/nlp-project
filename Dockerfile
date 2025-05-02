FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Instalar dependencias
# Install dependencies
RUN pip install --no-cache-dir transformers flask accelerate sentencepiece protobuf bitsandbytes
    # langchain langchain-huggingface langchain-community sentence-transformers langchain-chroma chromadb pypdf huggingface_hub[hf_xet]

# Copiar el código de la aplicación

COPY llm /app/llm

# Comando para ejecutar la aplicación
CMD ["python", "llm/llm_service.py"]