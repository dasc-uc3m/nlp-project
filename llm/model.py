from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import requests
from typing import Dict
import html
import os

class LocalLLM:
    def __init__(self):
        self.url: str = "http://localhost:5001"
        self.headers: Dict[str, str] = {"Content-Type": "application/json"}

    def __call__(self, data, stop=None) -> str:
        try:
            response = requests.post(self.url + "/generate", headers=self.headers, json={"messages": data})
            response.raise_for_status()
        except requests.HTTPError as e:
            print(e)
            return ""
        answer = response.json()["response"]
        return html.unescape(answer)

class CustomLLM:
    def __init__(
        self,
        model_name,
        device,
        torch_dtype,
        gen_kwargs
    ) -> None:
        self.model_loaded = False
        print(f"🚀 Starting to load model: {model_name} on {device}")
        self.model_name = model_name
        self.device = device

        # Load tokenizer
        print(f"🔑 Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=os.getenv("HUGGINGFACE_TOKEN", None)
        )
        print(f"✅ Tokenizer loaded successfully")

        # Load model
        print(f"🧠 Loading model {model_name}...")
        if torch_dtype in ["float32", "float16"]:
            dtype = torch.float32 if torch_dtype == "float32" else torch.float16
            dtype_config = {"torch_dtype": dtype}
        elif torch_dtype in ["int8", "int4"]:
            q_config = BitsAndBytesConfig(load_in_8bit=True) if torch_dtype == "int8" else BitsAndBytesConfig(load_in_4bit=True)
            dtype_config = {"quantization_config": q_config}
        else:
            print(f"Not supported dtype: {torch_dtype}. Defaulting to torch.float32.")
            dtype_config = {"torch_dtype": torch.float32}
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            # torch_dtype=torch.float16,
            # quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            **dtype_config,
            device_map={"": 0},
            token=os.getenv("HUGGINGFACE_TOKEN", None)
        )
        print(f"✅ Model loaded successfully and placed on {self.model.device}")

        # Generation parameters
        self.gen_kwargs = gen_kwargs
        if "max_new_tokens" in self.gen_kwargs and self.gen_kwargs["max_new_tokens"] == -1:
            self.gen_kwargs["max_new_tokens"] = self.model.config.max_position_embeddings
        
        # Identify model family
        self.model_family = self._identify_model_family()
        print(f"🔍 Model family identified as: {self.model_family}")

        # Check system role support
        self.supports_system = self._check_system_support()
        if self.supports_system:
            print(f"✅ Model supports system messages.")
        else:
            print(f"⚠️ Model does NOT support system messages. System prompts will be merged into user messages.")

        self.model_loaded = True
        print(f"🎉 Model {model_name} is ready to use!")

    def _identify_model_family(self):
        name = self.model_name.lower()
        if "qwen" in name:
            return "qwen"
        elif "gemma" in name:
            return "gemma"
        elif "llama" in name:
            return "llama"
        elif "mistral" in name:
            return "mistral"
        elif "vicuna" in name:
            return "vicuna"
        else:
            return "default"

    def _check_system_support(self):
        # Try applying a system prompt to see if the tokenizer supports it
        test_messages = [
            {"role": "system", "content": "Test system message"},
            {"role": "user", "content": "Test user message"}
        ]
        try:
            self.tokenizer.apply_chat_template(test_messages, tokenize=False)
            return True
        except Exception:
            return False

    def send_message(self, data):
        # Handle models that don't support system role (e.g., Gemma)
        if not self.supports_system:
            # Merge system messages into the first user message
            modified_data = []
            system_content = ""
            for message in data:
                if message.get("role") == "system":
                    system_content += message.get("content", "") + "\n"
                else:
                    modified_data.append(message)
            # Prepend system content to the first user message
            if system_content and modified_data:
                for i, message in enumerate(modified_data):
                    if message.get("role") == "user":
                        modified_data[i]["content"] = system_content.strip() + "\n\n" + message.get("content", "")
                        break
                # If no user message, add one
                if not any(m.get("role") == "user" for m in modified_data):
                    modified_data.insert(0, {"role": "user", "content": system_content.strip()})
            data = modified_data

        # Try to use the chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                text = self.tokenizer.apply_chat_template(
                    data,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"⚠️ Error applying chat template: {e}")
                text = self._manual_chat_template(data)
        else:
            text = self._manual_chat_template(data)

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            **self.gen_kwargs
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def _manual_chat_template(self, data):
        print(f"📝 Using manual chat template for {self.model_name}")
        if self.model_family == "gemma":
            # Gemma-specific template
            text = ""
            for message in data:
                role = message.get("role", "")
                content = message.get("content", "")
                if role == "user":
                    text += f"<start_of_turn>user\n{content}<end_of_turn>\n"
                elif role == "assistant":
                    text += f"<start_of_turn>model\n{content}<end_of_turn>\n"
            text += "<start_of_turn>model\n"
            return text
        else:
            # Generic template
            text = ""
            for message in data:
                role = message.get("role", "")
                content = message.get("content", "")
                if role == "system":
                    text += f"System: {content}\n\n"
                elif role == "user":
                    text += f"User: {content}\n\n"
                elif role == "assistant":
                    text += f"Assistant: {content}\n\n"
            text += "Assistant: "
            return text
