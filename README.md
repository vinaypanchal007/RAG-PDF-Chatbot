# RAG PDF Chatbot

A local multi-document RAG (Retrieval-Augmented Generation) chatbot built using LangChain, Ollama, Gemma, FAISS, and sentence-transformer embeddings.

The chatbot allows users to upload multiple PDF documents and ask questions based on their contents using semantic retrieval and local LLM inference.

---

# Features

- Multi-PDF support
- Semantic search using embeddings
- Local LLM inference using Ollama + Gemma
- Vector database using FAISS
- Context-aware question answering
- Offline AI workflow
- Retrieval-Augmented Generation (RAG)

---

# Tech Stack

## Backend
- Python
- LangChain

## LLM
- Ollama
- Gemma

## Vector Database
- FAISS

## Embeddings
- sentence-transformers
- all-MiniLM-L6-v2

## PDF Processing
- PyPDFLoader

---

# Project Architecture

```text
PDF Documents
      ↓
Document Loader
      ↓
Text Chunking
      ↓
Embeddings
      ↓
FAISS Vector Store
      ↓
Retriever
      ↓
Gemma LLM (Ollama)
      ↓
Answer Generation
