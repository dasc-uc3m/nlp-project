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

    result = chatbot.search_for_context(data, vector_db)
    return result+"\n"

@app.route("/upload_documents", methods=["POST"])
def upload_documents():
    global vector_db

    data = request.data.decode("utf-8")

    vector_db.load_document(data)
    return "Document loaded.\n"

@app.route("/infer", methods=["POST"])
def infer_with_chatbot():
    global chatbot

    data = request.data.decode("utf-8")

    answer = chatbot.infer(data)

    return jsonify({"response": answer})

app.run(host="0.0.0.0", port = int(os.environ.get("PORT", "5002")), debug=True)