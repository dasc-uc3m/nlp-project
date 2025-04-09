FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Instalar dependencias
RUN pip install --no-cache-dir transformers flask accelerate sentencepiece protobuf

# Copiar el código de la aplicación
COPY app/llm.py /app/
COPY src /app/src

# Comando para ejecutar la aplicación
CMD ["python", "llm.py"]