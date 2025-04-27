import streamlit as st
import requests
import json # Ensure json is imported
from sidebar import sidebar 

# API endpoint of the Flask service running in Docker
# It's accessible via localhost on the host port mapped in docker-compose (5001)
FLASK_API_URL = "http://localhost:5002/infer"



# Configure the page with a custom theme and layout
st.set_page_config(
    page_title="MidwifeGPT",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded"
)

sidebar()


# Custom CSS to improve the appearance
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        font-family: 'Arial', sans-serif;
    }
    .main-header {
        text-align: center;
        color: #ff69b4;
        padding: 2rem 0;
    }
    .chat-container {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Main title with custom styling
st.markdown("<h1 class='main-header'>🤰 MidwifeGPT</h1>", unsafe_allow_html=True)

# Add a welcoming message
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.markdown("""
        
            <h3>Welcome to MidwifeGPT! </h3>
            <p>I'm here to help answer your questions about pregnancy and childbirth.</p>
        </div>
    """, unsafe_allow_html=True)

# Create a container for the chat interface
chat_container = st.container()

with chat_container:
    # Display chat messages from history with improved styling
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])
            # Display sources if available
            if "sources" in message:
                with st.expander("View Sources"):
                    for idx, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        **Source {idx}:**  
                        File: `{source['source']}`  
                        Page: {source['page']}  
                        Preview: _{source['content']}_
                        ---
                        """)

    # Accept user input with a more prominent chat input
    if prompt := st.chat_input("Type your question here..."):
        # Add user message to history and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Prepare the payload for the Flask API
        api_payload = {"messages": st.session_state.messages}

        # Call the Flask API with improved error handling and user feedback
        try:
            with st.spinner("Thinking... 💭"): # More engaging spinner message
                response = requests.post(FLASK_API_URL, json=api_payload)
                response.raise_for_status()

                # Process the response
                api_response = response.json()
                assistant_message = api_response.get("response", "Error: No response received.")
                sources = api_response.get("sources", [])
                print(f"DEBUG - Received sources: {sources}")  # Add this line

                # Add assistant response to history and display
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": assistant_message,
                    "sources": sources  # Store sources in message history
                })
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(assistant_message)
                    if sources:
                        with st.expander("View Sources"):
                            for idx, source in enumerate(sources, 1):
                                st.markdown(f"""
                                **Source {idx}:**  
                                File: `{source['source']}`  
                                Page: {source['page']}  
                                Preview: _{source['content']}_
                                ---
                                """)

        except requests.exceptions.ConnectionError:
            st.error("Connection Error: Could not connect to the backend service. Please make sure the Docker container is running.")
        except requests.exceptions.RequestException as e:
            st.error(f"API Request Failed: {e}")
            try:
                st.error(f"Response details: {response.text}")
            except NameError:
                pass
        except json.JSONDecodeError:
            st.error(f"Failed to process the response: {response.text}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# # Add a footer
# st.markdown("""
#     <div style='text-align: center; padding: 20px; color: #666; font-size: 0.8em; position: fixed; bottom: 0; width: 70%; background-color: white;'>
#         Made by Carlos, Duarte, Sandra and Alex - Universidad Carlos III de Madrid  
#     </div>
# """, unsafe_allow_html=True)
