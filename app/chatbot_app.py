import sys
sys.path.append(".")
import os
from flask import Flask, request, jsonify
from src.chatbot import ChatBot
from src.db import VectorDB
import glob


DOCUMENTS_DIR = "data"  # Folder where the PDFs are stored
DOCUMENT_EXTENSIONS = ["*.pdf"]  # Supported document extensions

def create_app():
    """
    Factory function that creates and configures the Flask app.
    
    - Initializes ChatBot and VectorDB instances.
    - Loads documents into the vector store (if not already loaded).
    - Defines the available API routes for document upload, inference, listing, deletion, etc.
    
    Returns:
        Flask app instance
    """
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
        
    # Unset the context to leave the chatbot "clean"
    app.chatbot.remove_context()

    # These are flask routes, they can be used later in our streamlit app to allow us to directly use this from the frontend. right now, we are not using this
    # All backend logic is done in src inside wither chatbot or vector db. These are routes and we are only pointing the streamlit app to /infer on port 5002 
    
    @app.route("/upload", methods=["POST"])
    def upload_document():
        """
        Route: POST /upload
        Allows the user to upload a PDF document which is processed and added to the vector store.
        """
        try:
            if "file" not in request.files:
                return jsonify({"error": "No file provided"}), 400
            
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400
            
            if not file.filename.endswith(".pdf"):
                return jsonify({"error": "Only PDF files are supported"}), 400
            
            # Save the file temporarily
            temp_path = os.path.join("temp", file.filename)
            os.makedirs("temp", exist_ok=True)
            file.save(temp_path)
            
            print(f"DEBUG - Processing document: {file.filename}")
            # Upload to vector database
            app.vector_db.upload_document(temp_path)
            print(f"DEBUG - Document processed successfully: {file.filename}")
            
            # Clean up
            os.remove(temp_path)
            
            return jsonify({"message": "Document uploaded and processed successfully"})
            
        except Exception as e:
            print(f"Error in upload endpoint: {str(e)}")
            return jsonify({"error": str(e)}), 500
            

    @app.route("/infer", methods=["POST"])
    def infer_with_chatbot():
        """
        Route: POST /infer
        Takes a list of chat messages (from frontend), reconstructs conversation,
        retrieves context, and returns the chatbot's response.
        """
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
                        
            # Retrieve fresh context for the latest query if it isn't loaded yet.
            if not app.chatbot.has_context():
                print(f"DEBUG - Retrieving context for query: {latest_message}")
                app.chatbot.retrieve_context_from_db_with_reranking(latest_message, app.vector_db)
            
            answer, sources = app.chatbot.infer(latest_message)
            print(f"DEBUG - Sources from chatbot: {sources}")
            return jsonify({
                "response": answer,
                "sources": sources
            })
            
        except Exception as e:
            print(f"Error in infer endpoint: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/list_documents", methods=["GET"])
    def list_documents():
        """
        Route: GET /list_documents
        Returns a list of all loaded document filenames in the vector store.
        """
        try:
            documents = app.vector_db.list_documents()
            return jsonify({"documents": documents})
        except Exception as e:
            print(f"Error listing documents: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/delete_document", methods=["DELETE", "POST"])
    def delete_document():
        """
        Route: DELETE or POST /delete_document
        Removes a document and all its vectorized chunks from the store based on its filename.
        """
        try:
            data = request.get_json()
            filename = data.get("filename")
            
            if not filename:
                return jsonify({"error": "No filename provided"}), 400
            
            print(f"DEBUG - Attempting to delete document: {filename}")
            success = app.vector_db.delete_document(filename)
            
            if success:
                return jsonify({"message": f"Document {filename} deleted successfully"}), 200
            else:
                return jsonify({"error": f"Failed to delete document {filename}"}), 404
            
        except Exception as e:
            print(f"Error in delete_document endpoint: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/remove_context", methods=["DELETE", "POST"])
    def remove_context():
        """
        Route: DELETE or POST /remove_context
        Removes the current context loaded into the chatbot.
        """
        if app.chatbot.has_context():
            app.chatbot.remove_context()
            return jsonify({"message": "Context deleted succesfully"}), 200
        else:
            return jsonify({"message": "Chatbot has already no context loaded."}), 200
        
    @app.route("/reset_chatbot", methods=["DELETE", "POST"])
    def reset_chatbot():
        """
        Route: DELETE or POST /reset_chatbot
        Resets the chatbot's context and memory to default empty state.
        """
        if app.chatbot.has_context():
            app.chatbot.remove_context()
        app.chatbot.memory.reset_memory()
        return jsonify({"message": "Reset made succesfully."}), 200

    return app

# Create and run the application
if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5002")), debug=True)