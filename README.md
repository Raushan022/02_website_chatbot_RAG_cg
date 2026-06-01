# Website Chatbot RAG using LangChain, OpenAI and FAISS

## Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot that can answer questions from the content of a website.

The user provides a website URL, and the application:

1. Loads website content
2. Splits content into chunks
3. Generates embeddings
4. Stores embeddings in FAISS
5. Retrieves relevant chunks based on user questions
6. Uses an OpenAI LLM to generate answers from retrieved content

The chatbot answers questions using only the website content and displays source URLs for transparency.

---

## Features

- Load website content dynamically
- Chunk website text using RecursiveCharacterTextSplitter
- Generate embeddings using OpenAI Embeddings
- Store vectors in FAISS
- Persist FAISS index locally
- Reload existing vector database without regenerating embeddings
- Retrieve top relevant chunks using similarity search
- Answer questions using OpenAI LLM
- Display source URLs used for answering
- Prevent hallucinations using context-only prompting

---

## Project Architecture

```text
Website URL
      ↓
WebBaseLoader
      ↓
Documents
      ↓
Text Chunking
      ↓
Embeddings
      ↓
FAISS Vector Store
      ↓
Similarity Search
      ↓
Relevant Chunks
      ↓
Prompt Construction
      ↓
OpenAI LLM
      ↓
Answer + Sources
```

---

## Tech Stack

- Python
- LangChain
- OpenAI
- FAISS
- dotenv

---

## Installation

### Clone Repository

```bash
git clone <your-repository-url>
cd website-chatbot-rag
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate:

Windows

```bash
venv\Scripts\activate
```

Mac/Linux

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
```

---

## Run Project

```bash
python main.py
```

## Persisted Vector Store

The application stores FAISS indexes locally.

Benefits:

- Faster startup
- No repeated embedding generation
- Reduced OpenAI embedding cost
- Better user experience

Folder structure:

```text
vector_db/

├── website_1/
│   ├── index.faiss
│   └── index.pkl

├── website_2/
│   ├── index.faiss
│   └── index.pkl
```

---

## Learning Outcomes

Through this project I learned:

- RAG Architecture
- Document Loading
- Website Data Processing
- Text Chunking
- OpenAI Embeddings
- Vector Databases
- FAISS Similarity Search
- Metadata Handling
- Retrieval Pipelines
- Prompt Engineering
- Hallucination Reduction

---

## Author

Raushan Kumar

React Developer transitioning into GenAI Engineering and building hands-on RAG and LLM projects.
