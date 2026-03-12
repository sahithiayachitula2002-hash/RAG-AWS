# RAG Agent (Retrieval Augmented Generation)

## Overview

This project implements a **Retrieval-Augmented Generation (RAG) agent** that answers user queries using information retrieved from a document knowledge base. Instead of relying only on a language model, the system retrieves relevant document chunks from a vector database and provides them as context to generate accurate responses.

The goal of this project is to demonstrate how modern GenAI systems combine **LLMs, embeddings, and semantic search** to build intelligent document question-answering systems.

---

## Features

* Document ingestion and preprocessing
* Text chunking for efficient retrieval
* Embedding generation for semantic search
* Vector similarity search
* Context-aware response generation using an LLM
* Interactive query interface

---

## Tech Stack

* **Python**
* **FAISS** for vector similarity search
* **AWS Bedrock** for model inference
* **Titan Embeddings** for vector generation
* **Streamlit / CLI** for user interaction (depending on your implementation)

---

## Project Architecture

1. **Document Ingestion**
   Documents are loaded and split into smaller chunks.

2. **Embedding Generation**
   Each chunk is converted into a vector representation using an embedding model.

3. **Vector Storage**
   Embeddings are stored in a FAISS vector index.

4. **Query Processing**
   When a user asks a question, the system retrieves the most relevant chunks from the vector store.

5. **Response Generation**
   Retrieved context is passed to the LLM to generate a grounded answer.

---

## Installation

Clone the repository:

git clone https://github.com/yourusername/rag-agent.git

cd rag-agent

Install dependencies:

pip install -r requirements.txt

---

## Configuration

Create a `.env` file and add your credentials:

AWS_ACCESS_KEY=your_access_key
AWS_SECRET_KEY=your_secret_key
AWS_REGION=your_region

---

## Running the Project

Run the application:

python app.py

or (if using Streamlit)

streamlit run app.py

---

## Example Use Case

Users can ask questions about documents stored in the knowledge base, and the RAG agent will retrieve relevant information and generate accurate answers based on the retrieved content.
