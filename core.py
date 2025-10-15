# core.py

import os
from dotenv import load_dotenv

# Document Loading
from langchain_community.document_loaders import PyPDFLoader

# Text Splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings and Vector Stores
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LLM and Chains
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')

VECTOR_STORE_PATH = "vector_store.faiss"

def ingest_documents():
    """
    Ingests PDF documents, processes them, and stores them in a FAISS vector store
    using a local, open-source embedding model.
    """
    documents_path = "documents/"
    if not os.path.exists(documents_path) or not os.listdir(documents_path):
        return "Error: 'documents' folder is empty or not found."

    # --- CHANGED CODE BLOCK ---
    # Step 1: Get a list of all PDF filenames first.
    pdf_files = [f for f in os.listdir(documents_path) if f.endswith(".pdf")]
    if not pdf_files:
        return "Error: No PDF documents found to ingest."
    
    # Step 2: Get the correct count of the files.
    num_ingested_files = len(pdf_files)

    # Step 3: Load all pages from all the files.
    all_pages = []
    for filename in pdf_files:
        loader = PyPDFLoader(os.path.join(documents_path, filename))
        all_pages.extend(loader.load())
    # --------------------------

    # Now, continue processing using the loaded pages
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_pages)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)

    # Step 4: Use the correct file count in the success message.
    return f"Success! Ingested {num_ingested_files} documents."

def handle_query(query: str):
    """
    Handles a user query using the RAG pipeline with open-source models.
    """
    if not os.path.exists(VECTOR_STORE_PATH):
        return "Vector store not found. Please ingest documents first."

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()

    endpoint_llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        task="text-generation",
        max_new_tokens=512
    )

    llm = ChatHuggingFace(llm=endpoint_llm)

    prompt_template = """
    Answer the user's question based only on the following context.
    If the answer is not found in the context, respond with the exact phrase: "The context provided does not contain information on this topic."
    Do not add any other explanation or information.

    Context: {context}

    Question: {input}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": query})

    return response["answer"]
