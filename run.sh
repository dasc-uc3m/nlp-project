#!/bin/bash

# Path to your virtual environment
VENV_PATH="./.venv"
ACTIVATE="$VENV_PATH/bin/activate"

# Open Docker Compose in new terminal with venv
gnome-terminal -- bash -c "source $ACTIVATE && docker compose up --build; exec bash"

# Open Flask app in new terminal with venv
gnome-terminal -- bash -c "source $ACTIVATE && python3 app/chatbot_app.py; exec bash"

# Open Streamlit UI in new terminal with venv
gnome-terminal -- bash -c "source $ACTIVATE && streamlit run ui/streamlit_app.py; exec bash"

