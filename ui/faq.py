# flake8: noqa
import streamlit as st


@st.cache_data
def load_faq_md():
    return """
# FAQ
## How does MidWifeGPT work?
… long markdown …
"""

def faq():
    # When faq() is called, it pulls in the cached markdown and renders it.
    st.markdown(load_faq_md())
    