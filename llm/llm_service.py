import sys
sys.path.append(".")
import os
import json
from flask import Flask, request, jsonify
from llm.model import CustomLLM

app = Flask(__name__)

# Model mapping from frontend names to HuggingFace model names
MODEL_MAPPING = {
    "Gemma 3 1B": "google/gemma-3-1b-it",
    "Llama 3.2 1B": "meta-llama/Llama-3.2-1B",
    "Deepseek R1 Distill Qwen 1.5": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen 2.5 0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "Llama 3.2 3B": "meta-llama/Llama-3.2-3B-Instruct"
}

# Initialize with default model
current_model = os.getenv("MODEL_NAME", "google/gemma-2-2b-it")
generation_kwargs = json.loads(os.getenv("GENERATION_PARAMETERS", ""))
torch_dtype = os.getenv("DTYPE", "float32")
llm = CustomLLM(
    current_model,
    os.getenv("DEVICE", "cuda"),
    torch_dtype=torch_dtype,
    gen_kwargs=generation_kwargs
)

@app.route('/switch_model', methods=['POST'])
def switch_model():
    global llm, current_model
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({"error": "No model name provided"}), 400
            
        if model_name not in MODEL_MAPPING:
            return jsonify({"error": f"Model {model_name} not supported"}), 400
            
        # Get the HuggingFace model name
        hf_model_name = MODEL_MAPPING[model_name]
        
        # Only switch if it's a different model
        if hf_model_name != current_model:
            print(f"Switching from {current_model} to {hf_model_name}")
            # Create new LLM instance
            llm = CustomLLM(
                hf_model_name,
                os.getenv("DEVICE", "cuda"),
                int(os.getenv("MAX_TOKENS", -1)),
                float(os.getenv("TEMPERATURE", 0.7))
            )
            current_model = hf_model_name
            
        return jsonify({
            "message": f"Model switched to {model_name}",
            "model_name": model_name,
            "hf_model_name": hf_model_name
        })
        
    except Exception as e:
        print(f"Error switching model: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    global llm
    
    # Verificar que hay un modelo cargado
    if llm is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Obtener el prompt del request
    data = request.get_json()
    if not data or "messages" not in data:
        return jsonify({"error": "No 'messages' provided"}), 400

    messages = data["messages"]
    
    try:
        response = llm.send_message(messages)
        return jsonify({"response": response})
    
    except Exception as e:
        print(f"Exception inside /generate: {e}")
        return jsonify({"error": str(e), "data": data}), 500

@app.route('/health', methods=['GET'])
def health():
    global llm
    return jsonify({
        "status": "ok", 
        "model_loaded": llm is not None,
        "model_name": llm.model_name if llm else None,
        "device": llm.device if llm else None
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)