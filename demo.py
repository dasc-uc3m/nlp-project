import subprocess
import requests
import time

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

    vector_db.upload_document("data/hypertension.pdf")
    chatbot.retrieve_context_from_db("Documents related with hypertension issues", vector_db, k=3)

    answer = chatbot.infer("Tell me something about the provided context.")
    print(answer)
    answer = chatbot.infer("What did you say in the previoius answer.")
    print(answer)

if __name__ == "__main__":
    initialize_app()
    main()