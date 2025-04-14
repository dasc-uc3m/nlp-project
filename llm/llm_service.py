import sys
sys.path.append(".")
import os
from flask import Flask, request, jsonify
from llm.model import CustomLLM

app = Flask(__name__)
llm = CustomLLM(
    os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
    os.getenv("DEVICE", "cuda"),
    int(os.getenv("MAX_TOKENS", 512)),
    float(os.getenv("TEMPERATURE", 0.7))
)

@app.route('/generate', methods=['POST'])
def generate():
    global llm
    
    # Verificar que hay un modelo cargado
    if llm is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Obtener el prompt del request
    data = request.json
    # if not data or 'prompt' not in data:
    #     return jsonify({"error": "No prompt provided"}), 400
    
    prompt = data["prompt"]
    
    try:
        
        # context = retrieve_context(prompt, k = 3)
        
        # rag_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
        
        # Generar respuesta
        messages = [
            {"role": "system", "content": "You are a helpful ChatBot assistant that provide information given certain context."},
            {"role": "user", "content": prompt}
        ]
        response = llm.send_message(messages)
        
        
        return jsonify({"response": response})
    
    except Exception as e:
        return jsonify({"error": str(e),
                        "data": data,
                        "request": request}), 500

@app.route('/health', methods=['GET'])
def health():
    global llm
    return jsonify({
        "status": "ok", 
        "model_loaded": llm is not None,
        "model_name": llm.model_name if llm else None,
        "device": llm.device if llm else None
    })

app.run(host="0.0.0.0", port = int(os.environ.get("PORT", "5000")), debug=True)