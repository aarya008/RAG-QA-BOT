import torch
import os
import re
import math
import hashlib
import time
import shutil # Added for directory removal
from typing import Tuple, Any, Literal
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import gradio as gr



# Constants
MAX_FILE_SIZE_MB = 100
MAX_PAGES = 200
PERSIST_DIRECTORY = "db"

def generate_file_hash(file_path: str, chunk_size: int = 5000) -> str:
    """
    Generates a SHA-256 hash for a given file.

    Args:
        file_path (str): The path to the file.
        chunk_size (int): The size of chunks to read from the file.

    Returns:
        str: The hexadecimal representation of the file's SHA-256 hash.
    """
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

def get_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(8192)  # Read in 8KB chunks
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

## LLM
def get_llm():
    GOOGLE_API_KEY="AIzaSyAjFwYvWGe9efXx4RnqSfQXbA6Iz_zDkj8"
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, api_key =GOOGLE_API_KEY)

## Embedding
def get_embedding_model():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    # Detect CUDA/ROCm availability and set device accordingly
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[INFO] Using device: {device}")
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': device})


## Document loader
def document_loader(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


## Text splitter
def text_splitter(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


## Load and split docs
def load_and_split_docs(file_path):
    documents = document_loader(file_path)
    chunks = text_splitter(documents)
    return chunks


## Vector db
def _create_and_persist_vectordb(chunks, embedding_model, persist_path):
    print(f"[INFO] Creating new vector store at {persist_path}")
    vectordb = Chroma.from_documents(chunks, embedding_model, persist_directory=persist_path)
    vectordb.persist()
    print(f"[INFO] Vector store created and persisted.")
    return vectordb

def _load_existing_vectordb(persist_path, embedding_model):
    print(f"[INFO] Loading existing vector store from {persist_path}")
    vectordb = Chroma(persist_directory=persist_path, embedding_function=embedding_model)
    retriever_obj = get_retriever(vectordb)
    print(f"[INFO] Vector store loaded.")
    return retriever_obj, vectordb

def vector_database(file_path):
    file_hash = get_file_hash(file_path)
    persist_path = os.path.join(PERSIST_DIRECTORY, file_hash)
    embedding_model = get_embedding_model()

    if os.path.exists(persist_path) and all(os.path.getsize(os.path.join(persist_path, f)) > 0 for f in os.listdir(persist_path)):
        retriever_obj, vectordb = _load_existing_vectordb(persist_path, embedding_model)
    else:
        docs = load_and_split_docs(file_path)
        retriever_obj, vectordb = _create_new_vectordb(docs, embedding_model, persist_path)

    return retriever_obj, vectordb


## Retriever
def get_retriever(vectordb):
    return vectordb.as_retriever(search_kwargs={"k": 20})


## RAG
# Gradio upload function
def _validate_file(file_hash):
    persist_directory = os.path.join(PERSIST_DIRECTORY, file_hash)
    return os.path.exists(persist_directory) and os.listdir(persist_directory)



def _create_new_vectordb(docs, embedding_model, persist_dir):
    print(f"[INFO] Creating new vector database at {persist_dir}")
    vectordb = _create_and_persist_vectordb(docs, embedding_model, persist_dir)
    retriever_obj = get_retriever(vectordb)
    return retriever_obj, vectordb

def upload_file(file_path, progress_callback=None):
    if file_path is None:
        return "Please upload a PDF file.", None, None

    if not os.path.isfile(file_path):
        return "Invalid file path. Please provide a valid PDF file.", None, None

    if not os.path.splitext(file_path)[1].lower() == '.pdf':
        return "Invalid file type. Please upload a PDF file.", None, None

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return f"File size exceeds {MAX_FILE_SIZE_MB} MB.", None, None

    file_hash = get_file_hash(file_path)
    persist_dir = os.path.join(PERSIST_DIRECTORY, file_hash)

    if _validate_file(file_hash):
        embedding_model = get_embedding_model()
        retriever_obj, vectordb = _load_existing_vectordb(persist_dir, embedding_model)
        return "Existing vector database loaded.", retriever_obj, vectordb
    else:
        retriever_obj, vectordb = vector_database(file_path)
        return "New vector database created.", retriever_obj, vectordb

def _get_trimmed_context(docs, query: str, max_context_length: int) -> str:
    context = "\n".join([doc.page_content for doc in docs])
    if len(context) + len(query) > max_context_length:
        context = context[:max_context_length - len(query) - 50] + "... (context truncated)"
        print(f"[INFO] Context truncated due to length. Original length: {len(context) + len(query)}, Truncated length: {len(context)}")
    return context

def retriever_qa(query: str, history: list, retriever_obj, vectordb) -> Tuple[list, str]:
    if retriever_obj is None:
        return (history + [(query, "Please upload a PDF file first.")], "Error: No file uploaded")

    llm = get_llm()

    # Define a maximum context length to prevent exceeding model limits
    MAX_CONTEXT_LENGTH = 6000  # Adjust this value based on model limitations and desired performance. For more accuracy, a token counter should be used.

    try:
        start_time = time.time()
        # Get relevant documents
        docs = retriever_obj.invoke(query)
        print(f"[INFO] Document retrieval took {time.time() - start_time:.2f} seconds.")

        # Create context from retrieved documents
        context = _get_trimmed_context(docs, query, MAX_CONTEXT_LENGTH)

        # Prompt with context and question
        system_prompt = """
        You are a helpful AI assistant. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know. Do not try to make up an answer.
        Cite the source page numbers or excerpts from the context if available.
        """
        prompt = f"{system_prompt}\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"

        start_time = time.time()
        # Get response from LLM
        response = llm.invoke(prompt)
        filtered_response = re.sub(r'\*\*(.*?)\*\*', r'\1', response.content)
        print(f"[INFO] LLM invocation took {time.time() - start_time:.2f} seconds.")

        # Update chat history
        history.append((query, filtered_response))
        return (history, "QA process complete.")
    except Exception as e:
        error_message = f"Error during QA: {e}"
        print(f"[ERROR] {error_message}")
        history.append((query, error_message))
        return (history, error_message)


def get_qa_chain(retriever_obj):
    llm = get_llm()
    qa_chain = (
        {"context": retriever_obj, "question": RunnablePassthrough()} | llm | StrOutputParser()
    )
    return qa_chain


def delete_vectordb(file_hash):
    persist_directory = os.path.join(PERSIST_DIRECTORY, file_hash)
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"Deleted existing vector database for hash: {file_hash}")


# Gradio wrapper for file upload
def gradio_upload_file(file, retriever_state, vectordb_state):
    if file is None:
        return "Please upload a PDF file.", None, None, []
    
    file_path = file.name

    # Clean up previous vector database if it exists
    if vectordb_state is not None and hasattr(vectordb_state, 'file_hash'):
        delete_vectordb(vectordb_state.file_hash)

    status_message, retriever_obj, vectordb = upload_file(file_path)
    if vectordb is not None:
        vectordb.file_hash = get_file_hash(file_path) # Store file_hash in vectordb for cleanup
    return status_message, retriever_obj, vectordb, []

# Gradio wrapper for QA
def gradio_qa(query, history, retriever_state):
    if retriever_state is None:
        return "", history + [(query, "Please upload a PDF file first.")], "Error: No file uploaded"

    updated_history, status_message = retriever_qa(query, history, retriever_state, None)
    return "", updated_history, status_message

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# RAG chat bot")

    retriever_state = gr.State(None)
    vectordb_state = gr.State(None)

    with gr.Row(): # Main Row
        with gr.Column(scale=1): # Left Column for Upload and Query
            with gr.Row(): # Row for Document Upload
                file_component = gr.File(label="Upload PDF", file_types=[".pdf"])
                upload_status_textbox = gr.Textbox(label="Upload Status", interactive=False)
            with gr.Row(): # Row for Input Query
                question_textbox = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
                with gr.Column(scale=0): # Column for buttons next to textbox
                    submit_button = gr.Button("Submit")
                    clear_button = gr.Button("Clear")
        with gr.Column(scale=2): # Right Column for Chatbot
            chatbot_component = gr.Chatbot(label="Conversation", height=450)

    file_component.upload(
        gradio_upload_file,
        inputs=[file_component, retriever_state, vectordb_state],
        outputs=[upload_status_textbox, retriever_state, vectordb_state, chatbot_component]
    )

    submit_button.click(
        gradio_qa,
        inputs=[question_textbox, chatbot_component, retriever_state],
        outputs=[question_textbox, chatbot_component, upload_status_textbox]
    )

    clear_button.click(
        lambda retriever_state, vectordb_state: ([], "", None, None, "", delete_vectordb(vectordb_state.file_hash) if vectordb_state and hasattr(vectordb_state, 'file_hash') else None),
        inputs=[retriever_state, vectordb_state],
        outputs=[chatbot_component, question_textbox, retriever_state, vectordb_state, upload_status_textbox]
    )

demo.launch(share=False, server_name="127.0.0.1", server_port=7860)