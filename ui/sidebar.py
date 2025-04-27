import streamlit as st
from faq import faq


def sidebar():
    
    st.sidebar.image("images/logo.png", use_container_width =True)
    
    st.sidebar.title("How to use")
    st.sidebar.markdown(
        "1. Ask a question💬\n"

    )
    # This line actually runs the code in faq.py:
    with st.sidebar.expander("ℹ️ FAQ"):
        faq()             # ← calls the function from faq.py