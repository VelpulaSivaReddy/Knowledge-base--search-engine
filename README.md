# 📚 Knowledge-base Search Engine

This is a web application that allows you to chat with your PDF documents. It uses a Retrieval-Augmented Generation (RAG) pipeline to provide synthesized answers based on the content of the uploaded files. The entire project runs using free, open-source models, requiring no paid API keys.


---

## ✨ Features

-   **File Upload:** Upload one or more PDF documents directly through the web interface.
-   **Document Ingestion:** Process and index the text content of your PDFs to create a searchable knowledge base.
-   **Q&A Interface:** Ask questions in a simple chat interface and receive answers generated from the document's content.
-   **Local Embeddings:** Uses `sentence-transformers` to create text embeddings locally on your machine, free of charge.
-   **Open-Source LLM:** Leverages the Hugging Face Inference API to generate answers using powerful models like Mixtral.

---

## 🛠️ Tech Stack

-   **Backend:** Python, LangChain
-   **Frontend:** Streamlit
-   **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
-   **Vector Store:** FAISS (Facebook AI Similarity Search)
-   **LLM:** Hugging Face Inference API (`mistralai/Mixtral-8x7B-Instruct-v0.1`)

---

## 🚀 Setup and Installation

Follow these steps to run the project locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/VelpulaSivaReddy/Knowledge-base--search-engine.git](https://github.com/VelpulaSivaReddy/Knowledge-base--search-engine.git)
cd Knowledge-base--search-engine 


### 2. Create a Virtual Environment
Create and activate a virtual environment to keep your project dependencies isolated.

**For Windows:**
```bash
python -m venv .venv
.\.venv\Scripts\activate

For macOS/Linux:
Bash
python3 -m venv .venv
source .venv/bin/activate



3. Install Dependencies
Install all the required Python libraries using the requirements.txt file.

Bash

pip install -r requirements.txt


### 4. Set Up Environment Variables
Create a file named `.env` in the root of your project folder. In this file, you need to add your Hugging Face API token. You can get a free token with **"Write"** permissions from the Hugging Face website.

HUGGINGFACEHUB_API_TOKEN="hf_...your_token_here"



How to Run the Application
With your virtual environment activated and dependencies installed, run the following command in your terminal:

Bash

streamlit run app.py
