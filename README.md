# 🌌 Aether AI: Conversational RAG Assistant

<div align="center">

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-FF9900.svg)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-purple.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

*A production-grade, locally-persisted Corrective & Self-Reflective Conversational RAG pipeline developed to answer complex queries strictly from company documents.*

</div>

---

## 🪐 Project Overview

Standard Retrieval-Augmented Generation (RAG) systems frequently struggle in production due to three core challenges: **retrieval noise** (injecting irrelevant text), **hallucinations** (unsupported model outputs), and **high latency** (processing simple tasks through heavy pipelines).

This project solves these exact issues by employing a highly efficient **Dual-Path processing topology**:

1. **The Fast Path (Low-Latency Bypass)**: Simple inputs, greetings, or direct conversation bypass heavy vector stores entirely using an in-memory embedding-based **Semantic Router**, routing queries to a lightweight conversational agent in milliseconds.
2. **The Heavy Path (Self-Reflective Agents)**: Complex queries are routed to specialized pipelines where Pydantic Query Analyzers parse constraints. Retrieval combines dense (FAISS) and sparse (BM25) search indices via Reciprocal Rank Fusion (RRF), followed by a multi-agent **LangGraph** self-correcting loop. Final answers are strictly validated against hallucinations.

---

## 🏗️ Detailed System Architecture

Aether AI is built on a highly modular, multi-layered architecture designed for low latency, strict data grounding, and scalable agentic workflows. 

```mermaid
graph TD
    %% Styling Definitions
    classDef client fill:#eef2ff,stroke:#6366f1,stroke-width:2px,color:#1e1b4b;
    classDef api fill:#f0fdf4,stroke:#22c55e,stroke-width:2px,color:#14532d;
    classDef router fill:#fff7ed,stroke:#f97316,stroke-width:2px,color:#7c2d12;
    classDef db fill:#fdf4ff,stroke:#d946ef,stroke-width:2px,color:#701a75;
    classDef agent fill:#ecfdf5,stroke:#10b981,stroke-width:2px,color:#064e3b;
    classDef external fill:#f8fafc,stroke:#64748b,stroke-width:2px,color:#0f172a,stroke-dasharray: 5 5;

    %% 1. Client & API Layer
    Client[🖥️ Client / Web UI]:::client
    API[🚀 FastAPI Backend]:::api
    Client -->|REST / JSON| API

    %% 2. Orchestration & Routing
    subgraph Routing
        Router{🔀 Semantic Router}:::router
        API --> Router
    end

    %% 3. Fast Path
    subgraph Fast Path
        FastChain[⚡ Fast LLM Chain]:::agent
        Router -->|Simple / Greetings| FastChain
    end

    %% 4. Heavy Path: Analysis & Hybrid Retrieval
    subgraph Heavy Path Analysis & Retrieval
        QueryAnalyzer[🔬 Pydantic Query Analyzer]:::agent
        Router -->|Complex Queries| QueryAnalyzer
        
        FAISS[(FAISS Dense Vectors)]:::db
        BM25[(BM25 Sparse Keywords)]:::db
        
        QueryAnalyzer --> FAISS
        QueryAnalyzer --> BM25
        
        RRF[🔀 Reciprocal Rank Fusion]:::api
        FAISS --> RRF
        BM25 --> RRF
        
        Reranker[🎯 Flashrank CPU Reranker]:::api
        RRF --> Reranker
    end

    %% 5. LangGraph Agentic Loop
    subgraph LangGraph Multi-Agent Workflows
        CRAG[🕸️ Corrective RAG Loop]:::agent
        Decomp[🌲 Decomposition Graph]:::agent
        
        Reranker -->|Standard Route| CRAG
        Reranker -->|Multi-Hop Route| Decomp
        
        Generator[🤖 Answer Generator]:::agent
        Grader{⚖️ Hallucination Grader}:::router
        DDG[🌐 DuckDuckGo Web Search]:::external
        
        CRAG --> Generator
        Generator --> Grader
        Grader -->|Fails Verification| DDG
        DDG --> Generator
        Grader -->|Passes Verification| Verified[✅ Verified Output]:::api
        Decomp --> Verified
    end

    %% Final Resolution
    FastChain --> Output[📤 Final Response]:::api
    Verified --> Output
    Output --> API
    
    %% External Services
    LLM((OpenAI / Cohere API)):::external
    FastChain -.- LLM
    QueryAnalyzer -.- LLM
    Generator -.- LLM
    Grader -.- LLM
```

### 🔹 Layer-by-Layer Breakdown

1. **Client & API Layer (FastAPI)**: Serves multiple endpoints (`/api/chat`, `/api/upload`, `/api/ingest`, `/api/config`). Handles CORS, payload validation, and serves the static dashboard.
2. **Orchestration & Routing (Semantic Router)**: Immediately evaluates the query against predefined semantic boundaries. If the query is conversational (e.g., "Hello", "Thanks"), it bypasses the database entirely, reducing API cost and latency to `< 500ms`.
3. **Query Analysis (Pydantic & LLM)**: For complex queries, a structured LLM extracts constraints (e.g., `publish_year > 2022`, `file_type: PDF`). These constraints are transformed into strict metadata filters for the vector stores.
4. **Hybrid Retrieval & Reranking Engine**: 
   - **FAISS (Dense)**: Retrieves documents conceptually related to the query.
   - **BM25 (Sparse)**: Ensures exact keyword matches (vital for serial numbers or acronyms).
   - **RRF & Flashrank**: Combines both streams via Reciprocal Rank Fusion and re-ranks them locally on the CPU using cross-encoder models, ensuring only the highest-fidelity context reaches the agent.
5. **Agentic Workflows (LangGraph)**:
   - **Corrective RAG (CRAG)**: Generates a draft answer and grades it for hallucinations. If the draft contains ungrounded claims, it dynamically triggers DuckDuckGo web search to gather missing facts, re-writes the context, and tries again.
   - **Decomposition**: For multi-faceted questions, the graph recursively breaks the problem into sub-questions, answering them sequentially before synthesizing a final response.

---

## 🔄 Application Flow (Request Lifecycle)

```mermaid
sequenceDiagram
    participant User
    participant Router as Semantic Router
    participant DB as Vector Stores
    participant Agent as LangGraph Agent
    participant LLM as Ollama (local)
    
    User->>Router: What is the Q1 revenue?
    Router-->>Router: Computes Cosine Similarity
    
    alt is Simple Query
        Router->>LLM: Direct Chat Prompt
        LLM-->>User: Immediate Response (< 500ms)
    else is Complex Query
        Router->>DB: Extract Metadata & Execute Hybrid Search
        DB-->>Agent: Return Top-K Chunks
        
        loop Corrective RAG (CRAG)
            Agent->>LLM: Generate Answer based on Chunks
            LLM-->>Agent: Draft Answer
            Agent->>LLM: Grade Answer for Hallucinations
            
            alt Passes Verification
                LLM-->>User: Fully Grounded Answer
            else Fails Verification
                Agent->>Agent: Trigger Web Search Fallback
                Agent->>DB: Re-write Query & Re-Retrieve
            end
        end
    end
```

---

## ⚡ Key Features

* 🔮 **Dual-Routing Switchboard**: Supports embedding-based local **Semantic Routing** (zero-token latency, in-memory) to classify query intent instantly.
* 🔬 **Pydantic Query Analyzer**: Automatically extracts database metadata filters (e.g., `publish_year`, `file_type`, `page_number`) to build strict queries.
* 🔀 **Hybrid Retrieval (RRF)**: Combines dense vector retrieval (FAISS) and sparse keyword retrieval (BM25) using Reciprocal Rank Fusion to ensure both semantic capture and exact keyword matches.
* 🕸️ **LangGraph Multi-Hop Decomposition**: Breaks complex, multi-faceted questions into sequential sub-questions, answering them one-by-one using intermediate context memory.
* 🌐 **Corrective RAG (CRAG)**: Grades retrieved documents and dynamically triggers DuckDuckGo web search to gather missing facts when local context is insufficient.
* 🪞 **Double-Guardrail Self-RAG Evaluator**: Uses a two-step validation chain (Hallucination Grader + Answer Relevance Grader) to run verification loops.
* ✂️ **Semantic Chunking**: Identifies meaning-based boundaries by tracking semantic drift across adjacent sentences, preventing paragraph truncation.
* 🎯 **Flashrank CPU Reranking**: Re-ranks candidates locally on CPU using optimized quantized cross-encoder models.
* ☁️ **Supabase Cloud Sync (Phase 3)**: Automatic backup and cross-device sync of documents, FAISS indices, and encrypted user configuration to Supabase Storage, secured via strict Row-Level Security (RLS) ensuring strict tenant data isolation without service-role keys.

---

## 📂 Project Structure & Evolution

This repository contains both the final production-ready system and the original prototypes from the first phase of the development lifecycle:

```text
├── src/                  # (ACTIVE) Final Aether AI Engine (LangGraph, FAISS, FastAPI)
├── tests/                # (ACTIVE) Pytest suite for the final pipeline
├── data/                 # (ACTIVE) Master directory for raw PDFs and CSV files
├── documents/            # (ACTIVE) Runtime ingestion directory for FAISS vector generation
├── eval/                 # (ACTIVE) Golden JSON sets for faithfulness evaluation scoring
├── evaluate.py           # (ACTIVE) Custom evaluation harness to generate accuracy metrics
├── prototype_v1/         # (ARCHIVE) Original TF-IDF/ChromaDB basic scripts
│   ├── src/              # Obsolete basic retriever logic
│   ├── tests/            # Obsolete tests for basic logic
│   └── scripts/          # Obsolete manual data cleaning scripts
├── .github/              # CI/CD Workflows (tests active src/ and tests/)
└── Dockerfile            # Containerization for the active backend
```

> **Note**: The `prototype_v1/` directory contains legacy scripts from early development stages, preserved for historical context.

---

## 🛠️ Developer Setup & Experience

### 1. Environment Setup

Clone the repository and install dependencies using `uv` (a much faster pip alternative):

```bash
git clone https://github.com/yourusername/aether-ai.git
cd aether-ai

# Install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Create a `.env` file from the example:
```bash
cp .env.example .env
```

No API keys are required — the chat model and the embedding model both run locally
through [Ollama](https://ollama.com), and the reranker (Flashrank) runs on CPU.
Pull the two models once, then make sure the Ollama daemon is running:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```
Override `OLLAMA_MODEL` / `OLLAMA_EMBED_MODEL` in `.env` to swap models — no code change needed.

### 2. Running Data Ingestion
Populate the FAISS and BM25 vector databases by processing the documents in the `documents/` folder:

```bash
python -m backend.ingest
```

### 3. Starting the Server
Run the FastAPI backend server:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

### 4. Running Evaluations & Tests
To view the accuracy metrics and system faithfulness (verifying the jump from the flawed v1 to the fixed v2 golden set):

```bash
python evaluate.py
```
To run the automated CI pipeline tests:
```bash
pytest tests/
```

---

## 🐳 Docker Deployment

For clean isolation, the project includes a production-ready multi-stage `Dockerfile`.

```bash
# Build the image
docker build -t aether-ai .

# Run the container. No API keys — but point it at the Ollama daemon on the host,
# since `localhost` inside the container is the container itself.
docker run -p 8000:8000 \
    -e OLLAMA_BASE_URL="http://host.docker.internal:11434" \
    aether-ai
```
*(Note: Docker ignores the archived `prototype_v1` and raw `data` folders to keep the container lightweight and strictly focused on production).*

---

## 🔒 Security & Quality Assurances
* **Least Privilege (Docker)**: The `Dockerfile` operates under a non-root user (`appuser`).
* **Path Traversal Protection**: Uploaded filenames are strictly sanitized before being saved to the file system.
* **No Leaked Secrets**: All API keys are loaded strictly via `.env` variables and `os.getenv()`.

---

> **Built with passion to push the boundaries of Local Corrective RAG systems.**
