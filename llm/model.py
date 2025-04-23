from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
from typing import Dict
import html

class LocalLLM:
    def __init__(self):
        self.url: str = "http://localhost:5001"
        self.headers: Dict[str, str] = {"Content-Type": "application/json"}

    def __call__(self, data, stop=None) -> str:
        try:
            response = requests.post(self.url+"/generate", headers=self.headers, json={"messages": data})
            response.raise_for_status()
        except requests.HTTPError as e:
            print(e)
        answer = response.json()["response"]
        return html.unescape(answer)

class CustomLLM():
    def __init__(
            self,
            model_name,
            device,
            max_tokens,
            temperature=0.7
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype = "auto",
            device_map = "auto"
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )
        self.max_tokens = max_tokens
        self.temperature = temperature

    def send_message(self, data):
        text = self.tokenizer.apply_chat_template(
            data,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response