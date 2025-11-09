# 🧠 RAGChat Bot — PDF Question Answering using LangChain, Gemini & Gradio

### 🚀 An intelligent chatbot that reads your documents, understands your questions, and gives precise answers — powered by **Retrieval-Augmented Generation (RAG)** with **LangChain**, **Google Gemini**, and **HuggingFace Embeddings**.

---

## 🗂️ Project Overview

This project implements a **RAG-based chatbot** capable of answering user queries from uploaded **PDF documents**.
It combines **Large Language Models (LLMs)** with a **Vector Database (Chroma)** to retrieve and reason over document content, making it an efficient and scalable **Document Q&A System**.

**Tech Stack Used:**

* 🧩 **LangChain** — RAG pipeline & data orchestration
* 🪶 **Google Gemini (Generative AI)** — LLM for reasoning and answering
* 🤗 **HuggingFace Sentence Transformers** — for high-quality text embeddings
* 🧱 **ChromaDB** — vector store for semantic search and retrieval
* ⚙️ **Gradio** — frontend UI for chatbot interaction
* 🐍 **Python** (with PyTorch) — core programming and model integration

---

## ✨ Features

✅ Upload any **PDF file** up to 100 MB
✅ Automatically extracts and splits text into chunks
✅ Generates **semantic embeddings** using HuggingFace
✅ Stores document knowledge in a **persistent vector database**
✅ Retrieves contextually relevant information using **Chroma retriever**
✅ Provides accurate and **LLM-powered answers** using Google Gemini
✅ Clean and modern **Gradio chatbot interface**
✅ Built with scalability and GPU support (CUDA/MPS detection)

---

## 🧩 Project Architecture

📄 PDF Upload
⬇️
📚 Document Loader (LangChain)
⬇️
✂️ Text Splitter
⬇️
🔢 HuggingFace Embeddings
⬇️
💾 Chroma Vector Database (Persistent Storage)
⬇️
🔍 Retriever
⬇️
🧠 Google Gemini LLM
⬇️
💬 Gradio Frontend Chatbot
⬇️
👤 User Interaction


## 💡 Core Functionalities

| Function                | Description                                      |
| ----------------------- | ------------------------------------------------ |
| `upload_file()`         | Handles PDF uploads and vector DB persistence    |
| `vector_database()`     | Builds or loads existing embeddings for a file   |
| `retriever_qa()`        | Core RAG logic for contextual question-answering |
| `get_llm()`             | Initializes Google Gemini (Gemini 2.5 Flash)     |
| `get_embedding_model()` | Uses sentence-transformers/all-mpnet-base-v2     |
| `gradio_qa()`           | Connects Gradio frontend to backend logic        |

---

## 🧰 Installation & Setup

### 1️⃣ Clone this repository

```bash
git clone https://github.com/<your-username>/RAGChatBot.git
cd RAGChatBot
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # (Linux/Mac)
venv\Scripts\activate         # (Windows)
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add your API key

Edit `ragbot.py` and replace:

```python
GOOGLE_API_KEY = "YOUR_API_KEY"
```

or export it as an environment variable:

```bash
export GOOGLE_API_KEY="your_google_api_key"
```

### 5️⃣ Run the chatbot

```bash
python ragbot.py
```

Open the link in your terminal (`http://127.0.0.1:7860`) to chat with your bot.

---

## 🖼️ Demo Preview

*(Insert screenshots or GIFs of your chatbot interface here)*

---

## 🧠 Skills Demonstrated

* Retrieval-Augmented Generation (RAG)
* LangChain workflow engineering
* LLM integration (Google Gemini)
* Vector Databases (Chroma)
* Natural Language Processing (NLP)
* PDF text extraction and document parsing
* UI development with Gradio
* GPU-aware model optimization (PyTorch)

---

## 🧾 Example Usage

1. Upload your PDF (research paper, report, documentation, etc.)
2. Ask: *“Summarize chapter 2 in simple terms.”*
3. The chatbot retrieves relevant content and gives a **concise, contextual, and AI-generated answer**.

---

## ⚖️ License

This project is licensed under the **MIT License** — you’re free to use and modify it with attribution.
(You can switch to Apache-2.0 if you want stronger protection for derivative works.)

---

## 👨‍💻 Author

**Aarya Tagare**
🎓 Electrical Engineer | 💡 Passionate about AI, ML, and Generative AI | Exploring Agentic AI Systems & Intelligent Automation 
📍 Kolhapur, India
🔗 [LinkedIn](www.linkedin.com/in/aary-tagare14) • [GitHub](https://github.com/aarya008) • [Email](mailto:tagareaary@gmail.com)

---

