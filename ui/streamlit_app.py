import streamlit as st
import requests
import pandas as pd
import time
import altair as alt

# ---- Constants ----
FLASK_API_URL = "http://localhost:5002/infer"
UPLOAD_URL = "http://localhost:5002/upload"
REFRESH_URL = "http://localhost:5002/refresh_documents"
LIST_DOCS_URL = "http://localhost:5002/list_documents"
DELETE_DOC_URL = "http://localhost:5002/delete_document"
DELETE_CONTEXT_URL = "http://localhost:5002/reset_chatbot"
MODEL_LIST = ["Llama 3.2 3B", "Gemma 3 1B", "Deepseek R1 Distill Qwen 1.5", "Qwen 2.5 0.5B"]
LLM_SERVICE_URL = "http://localhost:5001"

# Get current model from LLM service and reorder MODEL_LIST
try:
    response = requests.get(f"{LLM_SERVICE_URL}/health")
    if response.status_code == 200:
        current_model = response.json().get("model_name", "")
        # Find the frontend name for the current model
        for frontend_name, hf_name in {
            "Gemma 3 1B": "google/gemma-3-1b-it",
            "Deepseek R1 Distill Qwen 1.5": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "Qwen 2.5 0.5B": "Qwen/Qwen2.5-0.5B-Instruct"
        }.items():
            if hf_name == current_model:
                # Reorder MODEL_LIST to put current model first
                MODEL_LIST.remove(frontend_name)
                MODEL_LIST.insert(0, frontend_name)
                break
except Exception as e:
    print(f"Error getting current model: {str(e)}")

# ---- Page Config ----
st.set_page_config(
    page_title="MaternAI",
    page_icon="👩‍🍼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS ----
st.markdown("""
    <style>
    .stApp { max-width: 1200px; margin: 0 auto; font-family: 'Segoe UI', Arial, sans-serif; }
    .main-header { text-align: center; color: #ff69b4; padding: 2rem 0; font-size: 2.5rem;}
    .sidebar-title { color: #ff69b4; font-weight: bold; font-size: 1.2rem;}
    .doc-list { background: #fff0f6; border-radius: 8px; padding: 1rem; }
    .upload-section { background: #f5f5f5; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;}
    .chat-container { background: #f5f5f5; border-radius: 10px; padding: 20px; margin: 20px 0; }
    </style>
""", unsafe_allow_html=True)

# ---- Sidebar as Control Center ----
with st.sidebar:
    st.image("images/logo.png", width=100)
    st.markdown("## Welcome 👋")
    st.markdown("To use MaternAI, you can directly chat with the model, or upload your documents and ask questions about them.")
    st.markdown("You can also go to the document tab to manage your documents, or to the model comparison tab to see how different models perform on the same questions.")
    st.markdown("---")
    st.markdown("### Choose a LLM")
    st.markdown("If you want to experiment with different models, you can choose one from the dropdown menu.")
    st.markdown("These are all lightweight models that can run on a personal computer.")
    model: str = st.selectbox("Model", options=MODEL_LIST)  # type: ignore
    
    # Send model selection to backend
    if 'current_model' not in st.session_state:
        st.session_state.current_model = model
    
    if st.session_state.current_model != model:
        try:
            response = requests.post(
                f"{LLM_SERVICE_URL}/switch_model",
                json={"model_name": model}
            )
            if response.status_code == 200:
                st.success(f"Model switched to {model}")
                st.session_state.current_model = model
            else:
                st.error(f"Error switching model: {response.json().get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error connecting to LLM service: {str(e)}")
    
    st.markdown("---")
    st.markdown("**MaternAI v1.0**")
    st.markdown("MaternAI allows you to ask questions about your documents and get accurate answers with instant citations.")
    st.markdown("[Help & Docs](https://github.com/dasc-uc3m/nlp-project/tree/main)")

# ---- Main Header ----
st.markdown("<h1 class='main-header'> 👩‍🍼 MaternAI </h1>", unsafe_allow_html=True)

# ---- Tabs ----
tab1, tab2, tab3, tab4 = st.tabs(["🤖 Chat", "📄 Documents", "📊 Model Comparison", "📚 About"])

# ---- Chat Tab ----
with tab1:

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.markdown("""
            <h3>Welcome to MaternAI!</h3>
            <p>I'm here to help answer your questions about pregnancy and childbirth.</p>
        """, unsafe_allow_html=True)

    def display_message(message):
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for idx, source in enumerate(message["sources"], 1):
                        st.markdown(
                            f'''<div style="font-size:16px; color:#555;">
<strong>Source {idx}:</strong><br/>
File: <code>{source['source']}</code><br/>
Preview: <em>{source['content']}</em><br/>
<hr/>
</div>''',
                            unsafe_allow_html=True,
                        )

    def call_api(endpoint, payload=None, files=None):
        try:
            with st.spinner("Processing..."):
                if files:
                    response = requests.post(endpoint, files=files)
                else:
                    response = requests.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            st.error(f"API Request Failed: {e}")
            return None

    # Display existing chat history above the input
    for message in st.session_state.messages:
        display_message(message)

    # Place chat input below messages for follow-ups
    prompt = st.chat_input("Type your question here...")
    if prompt:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Call API
        api_payload = {"messages": st.session_state.messages}
        api_response = call_api(FLASK_API_URL, payload=api_payload)

        # Append assistant response if any
        if api_response:
            assistant_message = api_response.get("response", "Error: No response received.")
            sources = api_response.get("sources", [])
            st.session_state.messages.append({"role": "assistant", "content": assistant_message, "sources": sources})

        # Rerun to render updated messages above the input
        st.rerun()
    
        # Clear Chat button next to the chat
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.success("Chat cleared!")
        call_api(DELETE_CONTEXT_URL)
        st.rerun()

# ---- Documents Tab ----
with tab2:
    st.header("📤 Upload Documents")
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        if st.button("Upload"):
            with st.spinner("Uploading document..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(UPLOAD_URL, files=files)
                if response.status_code == 200:
                    success_message = st.success("Document uploaded successfully!")
                    time.sleep(2)
                    success_message.empty()
                else:
                    st.error(f"Error uploading document: {response.json().get('error', 'Unknown error')}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.header("📄 Documents in Database")
    try:
        response = requests.get(LIST_DOCS_URL)
        if response.status_code == 200:
            documents = response.json().get("documents", [])
            if documents:
                # Initialize session state for deletion if not exists
                if 'doc_to_delete' not in st.session_state:
                    st.session_state.doc_to_delete = None
                
                # Create a container for the delete confirmation
                delete_container = st.container()
                
                # Create a DataFrame with documents and add a delete button column
                df = pd.DataFrame([{"Name": doc} for doc in documents])
                df['Delete'] = False  # Add a column for delete buttons
                
                # Display the editable dataframe
                edited_df = st.data_editor(
                    df,
                    column_config={
                        "Delete": st.column_config.CheckboxColumn(
                            "Delete",
                            help="Select to delete this document",
                            default=False,
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Check if any document is selected for deletion
                for i, row in edited_df.iterrows():
                    if row['Delete']:
                        st.session_state.doc_to_delete = row['Name']
                        break
                
                # Show confirmation dialog if a document is selected for deletion
                if st.session_state.doc_to_delete:
                    with delete_container:
                        st.markdown("---")
                        st.warning(f"⚠️ You are about to delete: **{st.session_state.doc_to_delete}**")
                        st.markdown("This action cannot be undone.")
                        
                        col1, col2 = st.columns([1,2])
                        with col1:
                            if st.button("✅ Confirm Delete", type="primary"):
                                try:
                                    with st.spinner("Deleting document..."):
                                        print(f"DEBUG - Sending delete request for: {st.session_state.doc_to_delete}")
                                        del_resp = requests.post(DELETE_DOC_URL, json={"filename": st.session_state.doc_to_delete})
                                        print(f"DEBUG - Delete response status: {del_resp.status_code}")
                                        print(f"DEBUG - Delete response content: {del_resp.json()}")
                                        if del_resp.status_code == 200:
                                            st.success(f"✅ Successfully deleted: {st.session_state.doc_to_delete}")
                                            st.session_state.doc_to_delete = None
                                            time.sleep(1)
                                            st.rerun()
                                        else:
                                            st.error(f"❌ Error deleting document: {del_resp.json().get('error', 'Unknown error')}")
                                except Exception as e:
                                    print(f"DEBUG - Exception during deletion: {str(e)}")
                                    st.error(f"❌ Error: {str(e)}")
                        with col2:
                            if st.button("❌ Cancel", type="secondary"):
                                st.session_state.doc_to_delete = None
                                st.rerun()
            else:
                st.info("No documents in the database.")
        else:
            st.error(f"Error fetching documents: {response.json().get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        
        

# ---- Model Comparison Tab ----
with tab3:
    st.header("Model Comparison Report")
    st.markdown("""
    This section presents a comparison of AI models based on key metrics.
    Charts show average performance, while the table includes standard deviation.
    - **Retrieval metrics**: Recall@3, MRR
    - **BERTScore**: Precision, Recall, F1
    - **Generation speed**: Tokens/sec
    """)

    # --- Data Preparation ---

    # Model performance data (means)
    data = pd.DataFrame({
        "Model": [
            "Gemma-3-1b-it",
            "Qwen2.5-0.5B-Instruct",
            "DeepSeek-R1-Distill-Qwen-1.5B",
            "Llama-3.2-1B"
        ],
        "Short Name": ["Gemma-3", "Qwen2.5", "DeepSeek", "Llama-3.2"],
        "Recall@3": [0.558, 0.558, 0.558, 0.558],
        "MRR": [0.488, 0.488, 0.488, 0.488],
        "BS_P": [0.848, 0.835, 0.831, 0.875],
        "BS_R": [0.888, 0.890, 0.880, 0.894],
        "BS_F1": [0.867, 0.864, 0.854, 0.882],
        "Tokens/sec": [43.53, 67.13, 22.63, 32.24]
    })

    # Standard deviation data
    std_data = pd.DataFrame({
        "Model": [
            "Gemma-3-1b-it",
            "Qwen2.5-0.5B-Instruct",
            "DeepSeek-R1-Distill-Qwen-1.5B",
            "Llama-3.2-1B"
        ],
        "BS_P_std": [0.032, 0.019, 0.028, 0.031],
        "BS_R_std": [0.022, 0.021, 0.026, 0.025],
        "BS_F1_std": [0.024, 0.020, 0.024, 0.020],
        "Tokens/sec_std": [5.65, 18.37, 8.40, 2.31]
    })


    merged_data = pd.merge(data, std_data, on="Model", how="left")


    st.subheader("📊 Performance Charts")
    
    st.markdown("##### Retrieval Metrics")
    retrieval_chart_data = merged_data.set_index('Short Name')[['Recall@3', 'MRR']]
    st.bar_chart(retrieval_chart_data, height=300, use_container_width=True, stack=False) 

    st.markdown("##### Generation Speed")
    speed_chart_data = merged_data.set_index('Short Name')[['Tokens/sec']]
    st.bar_chart(speed_chart_data, height=300, use_container_width=True)

    st.markdown("##### BERTScore")
    
    bert_chart_data = merged_data.set_index('Short Name')[['BS_P', 'BS_R', 'BS_F1']]
    bert_chart_data.columns = ['Precision', 'Recall', 'F1 Score']
    st.bar_chart(bert_chart_data, height=300, use_container_width=True, stack=False)


    st.subheader("🔍 Detailed Performance Metrics")


    display_columns_ordered = [
        'Model', 'Short Name', 'Recall@3', 'MRR',
        'BS_P', 'BS_P_std', 'BS_R', 'BS_R_std', 'BS_F1', 'BS_F1_std',
        'Tokens/sec', 'Tokens/sec_std'
    ]
    display_data = merged_data[display_columns_ordered].copy()

    rename_map = {
        'BS_P': 'BS Precision', 'BS_P_std': 'BS Precision Std Dev',
        'BS_R': 'BS Recall', 'BS_R_std': 'BS Recall Std Dev',
        'BS_F1': 'BS F1', 'BS_F1_std': 'BS F1 Std Dev',
        'Tokens/sec': 'Avg Tokens/sec', 'Tokens/sec_std': 'Tokens/sec Std Dev'
    }
    display_data = display_data.rename(columns=rename_map)


    columns_to_highlight = [
        'Recall@3', 'MRR', 'BS Precision', 'BS Recall', 'BS F1', 'Avg Tokens/sec'
    ]

    highlight_style = 'background-color: lightgreen; font-weight: bold;'

    styler = display_data.style.highlight_max(
        subset=columns_to_highlight,
        axis=0, # axis=0 highlights the max in each column
        props=highlight_style
    ).format(
    
        {
            'Recall@3': "{:.3f}",
            'MRR': "{:.3f}",
            'BS Precision': "{:.3f}",
            'BS Precision Std Dev': "{:.3f}",
            'BS Recall': "{:.3f}",
            'BS Recall Std Dev': "{:.3f}",
            'BS F1': "{:.3f}",
            'BS F1 Std Dev': "{:.3f}",
            'Avg Tokens/sec': "{:.2f}",
            'Tokens/sec Std Dev': "{:.2f}"
        }
    )

    # Display the styled DataFrame
    st.dataframe(
        styler, # Pass the Styler object instead of the raw DataFrame
        use_container_width=True,
        hide_index=True
    )
    
    

    with st.expander("💡 Key Insights", expanded=True):
        fastest_model_idx = merged_data['Tokens/sec'].idxmax()
        best_bertscore_p_idx = merged_data['BS_P'].idxmax()
        best_bertscore_r_idx = merged_data['BS_R'].idxmax()
        best_bertscore_f1_idx = merged_data['BS_F1'].idxmax()

        fastest_model = merged_data.loc[fastest_model_idx]
        best_bertscore_p_model = merged_data.loc[best_bertscore_p_idx]
        best_bertscore_r_model = merged_data.loc[best_bertscore_r_idx]
        best_bertscore_f1_model = merged_data.loc[best_bertscore_f1_idx]

        st.markdown(f"""
        - **Speed Champion**: **{fastest_model['Short Name']}**
          (Avg: {fastest_model['Tokens/sec']:.1f} ± {fastest_model['Tokens/sec_std']:.1f} tokens/sec)
        - **BERTScore Leaders**:
          - Precision: **{best_bertscore_p_model['Short Name']}** (Avg: {best_bertscore_p_model['BS_P']:.3f} ± {best_bertscore_p_model['BS_P_std']:.3f})
          - Recall: **{best_bertscore_r_model['Short Name']}** (Avg: {best_bertscore_r_model['BS_R']:.3f} ± {best_bertscore_r_model['BS_R_std']:.3f})
          - F1: **{best_bertscore_f1_model['Short Name']}** (Avg: {best_bertscore_f1_model['BS_F1']:.3f} ± {best_bertscore_f1_model['BS_F1_std']:.3f})
        - **Retrieval Performance**: All models currently show identical average retrieval metrics
          (Recall@3: {merged_data['Recall@3'].iloc[0]:.3f}, MRR: {merged_data['MRR'].iloc[0]:.3f}).
        - **Trade-offs**: Consider the balance between generation speed (where **{fastest_model['Short Name']}** excels)
          and generation quality/similarity (where **{best_bertscore_f1_model['Short Name']}** leads in F1 score)
          based on specific application needs. Standard deviations in the table show result variability.
        """)


# ---- About Tab ----


with tab4:
    st.header("About this Project")
    st.markdown("""
    **MaternAI** is a project made by students at the University Carlos III de Madrid in the course of the subject of Natural Language Processing. 
    The main idea of the project is to create an AI assistant that can answer questions about pregnancy and childbirth using a custom made dataset of medical documents.
    The project is based on the RAG (Retrieval Augmented Generation) technique where the answer generated by the LLM is enriched with a set of relevant sources. 
    All models used are running locally, are open-source and free to use. Also we tried to make use of the lightest models available to run on a personal computer. 
    
    **Contacts:** 
    - Alejandro Merino - almebagar@gmail.com
    - Carlos Garijo - cgarijocrespo@gmail.com
    - Duarte Moura -  duartepcmoura@gmail.com 
    - Sandra Eizaguerri - 01eizaguerrisandra@gmail.com
    
    **GitHub:** [MaternAI](https://github.com/dasc-uc3m/nlp-project/tree/main)
    """)
