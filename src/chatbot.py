import sys
sys.path.append(".")

import requests
import html
from typing import Dict
from src.db import VectorDB

class LocalLLM:
    def __init__(self):
        self.url: str = "http://localhost:5001"
        self.headers: Dict[str, str] = {"Content-Type": "application/json"}

    def __call__(self, data, stop=None) -> str:
        try:
            response = requests.post(self.url+"/generate", headers=self.headers, json={"prompt": data})
            response.raise_for_status()
        except requests.HTTPError as e:
            print(e)
        answer = response.json()["response"]
        return html.unescape(answer)

class Memory:
    def __init__(self):
        self._memory = []
    
    def update_memory(self, human_msg: str, ai_msg: str):
        self._memory.append({"Human": human_msg})
        self._memory.append({"AI": ai_msg})

    def reset_memory(self):
        self._memory = []

    def compile(self):
        return "\n".join(f"{entity}: {message}" for message_dict in self._memory for entity, message in message_dict.items())


class ChatBot:
    def __init__(self):
        self.llm = LocalLLM()
        self.memory = Memory()
        self.default_prompt = (
            "You will be asked to take a conversation based on the following context:\n{context}\n"
            "--END OF CONTEXT--\n\n"
            "The following is the conversation that has been taken between a user and an AI assistant.\n"
            "{history}\n\n"
            "Human: {input}\nAI:"
        )
    
    def initialize_context(self, context: str):
        self.context = context

    def remove_context(self):
        del self.context

    def infer(self, message: str):
        if not hasattr(self, "context"):
            print("ERROR: You are asking the LLM but it hasn't got its context loaded.")
            return ""
            # prompt = self.default_prompt.format(context="", input=message, history=self.memory.compile())
        else:
            prompt = self.default_prompt.format(context=self.context, input=message, history=self.memory.compile())
        
        answer = self.llm(prompt)
        self.memory.update_memory(human_msg=message, ai_msg=answer)

        return answer

    def search_for_context(self, query, vector_db: VectorDB, k=3):
        context = vector_db.retrieve_context(query, k=k)
        if len(context) > 0:
            self.initialize_context(context=context)
            message = "Succesfully loaded context."
        else:
            message = "No context found in the Database."
        print(message)
        return message
        

    def __call__(self, message: str):
        self.infer(message=message)

if __name__=="__main__":
    cb = ChatBot()
    cb.infer("Hola")