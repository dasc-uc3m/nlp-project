import sys
sys.path.append(".")
import os
from flask import Flask, request, jsonify
from src.chatbot import ChatBot
from src.db import VectorDB

app = Flask(__name__)

chatbot = ChatBot()
vector_db = VectorDB()

@app.route("/search_document", methods=["POST"])
def search_for_documents():
    global chatbot
    global vector_db

    data = request.data.decode("utf-8")

    result = chatbot.retrieve_context_from_db(data, vector_db)
    return result+"\n"

@app.route("/upload_documents", methods=["POST"])
def upload_documents():
    global vector_db

    data = request.data.decode("utf-8")

    vector_db.upload_document(data)
    return "Document loaded.\n"

@app.route("/infer", methods=["POST"])
def infer_with_chatbot():
    global chatbot
    
    try:
        payload = request.get_json()
        messages = payload["messages"]
        latest_message = messages[-1]["content"]
        
        # Reset chatbot memory to sync with frontend
        chatbot.memory.reset_memory()
        
        # Rebuild memory from frontend history (excluding the last message)
        for msg in messages[:-1]:
            if msg["role"] in ["user", "assistant"]:
                if msg["role"] == "user":
                    user_msg = msg["content"]
                else:
                    chatbot.memory.update_memory(user_msg, msg["content"])
        
        answer = chatbot.infer(latest_message)
        return jsonify({"response": answer})
        
    except Exception as e:
        print(f"Error in infer endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

app.run(host="0.0.0.0", port = int(os.environ.get("PORT", "5002")), debug=True)