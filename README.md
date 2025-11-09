# 🧠 RAGChat Bot — PDF Question Answering using LangChain, Gemini & Gradio

### 🚀 An intelligent chatbot that reads your documents, understands your questions, and gives precise answers — powered by **Retrieval-Augmented Generation (RAG)** with **LangChain**, **Google Gemini**, and **HuggingFace Embeddings**.


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

## ✨ Features

✅ Upload any **PDF file** up to 100 MB
✅ Automatically extracts and splits text into chunks
✅ Generates **semantic embeddings** using HuggingFace
✅ Stores document knowledge in a **persistent vector database**
✅ Retrieves contextually relevant information using **Chroma retriever**
✅ Provides accurate and **LLM-powered answers** using Google Gemini
✅ Clean and modern **Gradio chatbot interface**
✅ Built with scalability and GPU support (CUDA/MPS detection)


## 🧩 Project Architecture

flowchart TD
A[PDF Upload] --> B[Document Loader (LangChain)]
B --> C[Text Splitter]
C --> D[HuggingFace Embeddings]
D --> E[Chroma Vector DB (Persistent)]
E --> F[Retriever]
F --> G[Google Gemini LLM]
G --> H[Gradio Frontend Chatbot]
H --> I[User Interaction]
```

---

## 💡 Core Functionalities

| Function                | Description                                      |
| ----------------------- | ------------------------------------------------ |
| `upload_file()`         | Handles PDF uploads and vector DB persistence    |
| `vector_database()`     | Builds or loads existing embeddings for a file   |
| `retriever_qa()`        | Core RAG logic for contextual question-answering |
| `get_llm()`             | Initializes Google Gemini (Gemini 2.5 Flash)     |
| `get_embedding_model()` | Uses sentence-transformers/all-mpnet-base-v2     |
| `gradio_qa()`           | Connects Gradio frontend to backend logic        |


## 🧠 Skills Demonstrated

* Retrieval-Augmented Generation (RAG)
* LangChain workflow engineering
* LLM integration (Google Gemini)
* Vector Databases (Chroma)
* Natural Language Processing (NLP)
* PDF text extraction and document parsing
* UI development with Gradio
* GPU-aware model optimization (PyTorch)

## 🧾 Example Usage

1. Upload your PDF (research paper, report, documentation, etc.)
2. Ask: *“Summarize chapter 2 in simple terms.”*
3. The chatbot retrieves relevant content and gives a **concise, contextual, and AI-generated answer**.


## ⚖️ License

This project is licensed under the **MIT License** — you’re free to use and modify it with attribution.
(You can switch to Apache-2.0 if you want stronger protection for derivative works.)

## 👨‍💻 Author

**Aarya Tagare**
🎓 Electrical Engineer | 💡 AI & Cybersecurity Enthusiast | ⚙️ Embedded Systems Developer
📍 Kolhapur, India
🔗 [LinkedIn](https://www.linkedin.com) • [GitHub](https://github.com/your-username) • [Email](mailto:your-email@example.com)

