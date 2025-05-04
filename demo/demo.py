import subprocess
import requests
import time
import os, sys

# añade la carpeta padre de demo/ al PYTHONPATH
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.chatbot import ChatBot
from src.db import VectorDB

def initialize_app():
    llm_exposed_port = 5001

    # This just checks if the LLM docker is already loaded, if not, it launches it.
    try:
        response = requests.get(f"http://localhost:{llm_exposed_port}/health")
        response.raise_for_status()
    except requests.ConnectionError:
        # If this exception happens, the LLM service app hasn't been launched.
        subprocess.run("docker compose up --build -d", shell=True, stdout=subprocess.DEVNULL)
        time.sleep(2)
        response = requests.get(f"http://localhost:{llm_exposed_port}/health")
        response.raise_for_status()

    if response.status_code != 200:
        # If status code is not 200 something is wrong.
        raise response
    else:
        print(response)
        print("Model is already loaded.")



def main():
    chatbot = ChatBot()
    vector_db = VectorDB()

    prompt = "¿Cuáles son los síntomas comunes asociados con el embarazo?"
    
    '''
    # ─── SIN QUERY EXPANSION ────────────────────────────────────────────────────────
    print("\n--- Without Query Expansion ---")
    # 1) Recuperar contexto simple (solo k documentos)
    _, _ = chatbot.retrieve_context_from_db(prompt, vector_db, k=3)

    print("Loaded context is:\n", chatbot.context if hasattr(chatbot, "context") else "<no context>")
    # 2) Inferencia con la query original
    answer_simple, _ = chatbot.infer(prompt, expand=False)
    print("Context used:")
    print(chatbot.context)
    print("\nAnswer:")
    print(answer_simple)
    '''
    # ─── CON QUERY EXPANSION + RE-RANKING ───────────────────────────────────────────
    print("\n--- With Query Expansion & Re-ranking ---")
    # 1) Recuperar contexto ampliado + re-ranking
    _, _ = chatbot.retrieve_context_from_db_with_reranking(
        prompt, vector_db, k_initial=10, k_final=5
    )

    # 4) Inferencia final con la misma query original
    answer_rerank, _ = chatbot.infer(prompt)
    print("Answer:")
    print(answer_rerank)

if __name__ == "__main__":
    initialize_app()
    main()