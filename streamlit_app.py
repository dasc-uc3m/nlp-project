import streamlit as st
import requests
import json # Ensure json is imported

# API endpoint of the Flask service running in Docker
# It's accessible via localhost on the host port mapped in docker-compose (5001)
FLASK_API_URL = "http://localhost:5002/infer"

st.title("Chat with LLM Service")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter your message:"):
    # Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare the payload for the Flask API
    # Your Flask app expects a JSON with a "messages" key containing the history [1]
    api_payload = {"messages": st.session_state.messages}

    # Call the Flask API
    try:
        with st.spinner("Waiting for response..."): # Optional spinner
            response = requests.post(FLASK_API_URL, json=api_payload)
            response.raise_for_status() # Check for HTTP errors

            # Process the response
            api_response = response.json()
            assistant_message = api_response.get("response", "Error: No response received.")

            # Add assistant response to history and display
            st.session_state.messages.append({"role": "assistant", "content": assistant_message})
            with st.chat_message("assistant"):
                st.markdown(assistant_message)

    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Could not connect to the backend service at {FLASK_API_URL}. Is the Docker container running?")
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Failed: {e}")
        # Optionally display more details from the response if available
        try:
            st.error(f"API Response: {response.text}")
        except NameError: # If response object doesn't exist
            pass
    except json.JSONDecodeError:
         st.error(f"Failed to decode API response. Response text: {response.text}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

