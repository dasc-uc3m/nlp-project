import requests
import html

from langchain_core.language_models import LLM

class LocalLLM(LLM):
    def __init__(self):
        self.url = "http://localhost:5000"
        self.headers = {"Content-Type": "application/json"}
    
    def _call(self, data) -> str:
        try:
            response = requests.post(self.url+"/generate", headers=self.headers, json=data)
            response.raise_for_status()
        except requests.HTTPError as e:
            print(e)
        answer = response.json()["response"]
        return html.unescape(answer)

    @property
    def _llm_type(self) -> str:
        return "local-llm"
