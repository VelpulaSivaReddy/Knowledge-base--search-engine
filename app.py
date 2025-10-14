# app.py

import streamlit as st
from core import ingest_documents, handle_query
import os
import shutil # Import the shutil library for directory operations

# --- Page Configuration ---
st.set_page_config(
    page_title="Knowledge-base Search",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Knowledge-base Search Engine")
st.write("""
This application allows you to search across your documents and get synthesized answers.
It uses a Retrieval-Augmented Generation (RAG) pipeline.
""")

# --- Sidebar for Document Ingestion ---
with st.sidebar:
    st.header(" Ingest Documents")
    st.write("Upload your PDF documents here. The system will process and index them.")

    uploaded_files = st.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True
    )

    if st.button("Ingest Documents"):
        if uploaded_files:
            documents_dir = "documents"
            
            # --- CHANGED CODE BLOCK ---
            # Step 1: Clean up the old documents directory before saving new ones.
            # This ensures that only the newly uploaded files are processed.
            if os.path.exists(documents_dir):
                shutil.rmtree(documents_dir)
            os.makedirs(documents_dir)
            # --------------------------

            # Step 2: Save the newly uploaded files to the clean directory.
            for uploaded_file in uploaded_files:
                with open(os.path.join(documents_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # Step 3: Run the ingestion process on the new files.
            with st.spinner("Processing documents... this may take a moment."):
                result = ingest_documents()
            st.success(result)
        else:
            st.warning("Please upload at least one PDF file.")

# --- Main Chat Interface ---
st.header(" Ask your Question")
st.write("Once your documents are ingested, you can ask questions below.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user",avatar="❓"):
        st.markdown(prompt)

    # Get the synthesized answer from the backend
    with st.spinner("Searching documents and synthesizing answer..."):
        response = handle_query(prompt)
        with st.chat_message("assistant",avatar="💡"):
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})