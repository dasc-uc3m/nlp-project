import sys
sys.path.append(".")
import os
from flask import Flask, request, jsonify
from src.chatbot import ChatBot
from src.db import VectorDB
import glob


# changes from before:
#     - now we have an application factory function create_app() - apparently its better for some stuff like starting up the document ingestion beforehand d

# Configuration
DOCUMENTS_DIR = "data"  
DOCUMENT_EXTENSIONS = ["*.pdf"]  

# def load_initial_documents():
#     """Load all documents from the documents directory"""
#     documents = []
#     for ext in DOCUMENT_EXTENSIONS:
#         documents.extend(glob.glob(os.path.join(DOCUMENTS_DIR, ext)))
#     return documents

def create_app():
    """Application factory function"""
    app = Flask(__name__)
    
    # Initialize chatbot and vector_db
    app.chatbot = ChatBot()
    app.vector_db = VectorDB()
    
    # Check if documents are already loaded
    collection_size = len(app.vector_db.vector_store.get()["ids"])
    
    if collection_size == 0:  # Only load if the collection is empty
        documents = glob.glob("data/*.pdf")
        print(f"Loading initial documents: {documents}")
        app.vector_db.upload_documents(documents)
        # Set initial context for chatbot
        app.chatbot.retrieve_context_from_db("general context", app.vector_db)
        print("Initial documents loaded successfully")
    else:
        print(f"Using existing collection with {collection_size} documents")
        app.chatbot.retrieve_context_from_db("general context", app.vector_db)



    # Why this is here!!
    #These are flask routes, they can be used later in our streamlit app to allow us to directly use this from the frontend. right now, we are not using this
    # All backend logic is done in src inside wither chatbot or vector db. These are routes and we are only pointing the streamlit app to /infer on port 5002 
    
    
    # # Define routes
    # @app.route("/refresh_documents", methods=["POST"])
    # def refresh_documents():
    #     documents = glob.glob("data/*.pdf")
    #     app.vector_db.upload_documents(documents)
    #     return jsonify({"message": "Documents refreshed successfully"})

    # @app.route("/search_document", methods=["POST"])
    # def search_for_documents():
    #     data = request.data.decode("utf-8")
    #     result, sources = app.chatbot.retrieve_context_from_db(data, app.vector_db)
    #     return jsonify({
    #         "message": result,
    #         "sources": sources
    #     })

    # @app.route("/upload_documents", methods=["POST"])
    # def upload_documents():
    #     try:
    #         data = request.data.decode("utf-8")
    #         app.vector_db.upload_document(data)
    #         # Update chatbot context after new document
    #         app.chatbot.retrieve_context_from_db("general context", app.vector_db)
    #         return "Document loaded successfully\n"
    #     except Exception as e:
    #         return jsonify({"error": str(e)}), 500

    @app.route("/infer", methods=["POST"])
    def infer_with_chatbot():
        try:
            payload = request.get_json()
            messages = payload["messages"]
            latest_message = messages[-1]["content"]
            
            # Reset chatbot memory to sync with frontend
            app.chatbot.memory.reset_memory()
            
            # Rebuild memory from frontend history (excluding the last message)
            for msg in messages[:-1]:
                if msg["role"] in ["user", "assistant"]:
                    if msg["role"] == "user":
                        user_msg = msg["content"]
                    else:
                        app.chatbot.memory.update_memory(user_msg, msg["content"])
            
            app.chatbot.retrieve_context_from_db(latest_message, app.vector_db)
            
            answer, sources = app.chatbot.infer(latest_message)
            print(f"DEBUG - Sources from chatbot: {sources}")
            return jsonify({
                "response": answer,
                "sources": sources
            })
            
        except Exception as e:
            print(f"Error in infer endpoint: {str(e)}")
            return jsonify({"error": str(e)}), 500

    return app

# Create and run the application
if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5002")), debug=True)