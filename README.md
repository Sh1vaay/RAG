# 🌌 Advanced Local Conversational RAG System

<div align="center">

![RAG System Header](https://raw.githubusercontent.com/Sh1vaay/RAG/main/header.png)

*An enterprise-ready, locally persisted Conversational RAG pipeline built on high-fidelity query optimizations, hybrid search indexes, and local cross-encoder re-ranking.*

&nbsp;

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/UV-Package%20Manager-00D2B4?style=for-the-badge&logo=cargo&logoColor=white)](https://github.com/astral-sh/uv)
[![Framework](https://img.shields.io/badge/LangChain-v1.0%2B-F15A24?style=for-the-badge&logo=chainlink&logoColor=white)](https://github.com/langchain-ai/langchain)
[![VectorDB](https://img.shields.io/badge/FAISS-VectorStore-blue?style=for-the-badge&logo=facebook&logoColor=white)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

</div>

---

## 🗺️ Table of Contents
* [🪐 Project Overview](#-project-overview)
* [⚡ Key Capabilities](#-key-capabilities)
* [📈 Architecture & Data Flow](#-architecture--data-flow)
  * [1. Ingestion Pipeline](#1-data-ingestion-pipeline)
  * [2. Multi-Query Retrieval & Fusion](#2-multi-query-hybrid-retrieval-pipeline)
  * [3. Conversational Lifecycle](#3-request-lifecycle--conversational-memory)
* [📊 Baseline RAG vs. Advanced RAG](#-baseline-rag-vs-advanced-rag)
* [🛠️ Tech Stack](#️-tech-stack)
* [🚀 Getting Started](#-getting-started)
  * [Configuration (`.env`)](#configuration-env)
  * [Installation](#installation)
  * [Usage](#usage)
* [🔒 Security & Optimization](#-security--optimization)
* [📈 Monitoring](#-monitoring--observability)
* [📄 License](#-license)

---

## 🪐 Project Overview

Standard RAG architectures frequently fail when handling multi-format datasets, breaking critical contexts, or failing to identify specific keyword matches. 

This project addresses these challenges by implementing an **offline, high-recall, high-precision retrieval pipeline**. It digests mixed formats (PDFs, CSVs, Word files, Text) locally, divides content using vector-similarity boundaries (Semantic Chunking), translates inputs to bypass poorly phrased queries, and uses a local Cross-Encoder to re-rank the context before generating a final answer.

> [!IMPORTANT]  
> **Privacy by Design**: All embeddings, databases, and re-ranking calculations are computed **locally**. No document text ever leaves your machine; API calls are strictly limited to LLM inference.

---

## ⚡ Key Capabilities

* 📂 **Multi-Format Processing**: Routes PDFs (`PyPDFLoader`), CSVs (`CSVLoader`), Word files (`Docx2txtLoader`), and text documents directly from a local `./documents` folder.
* 🔮 **Configurable Dual-Routing Engine**: Supports both a millisecond-level embedding-based **Semantic Router** (zero LLM token cost) and a structured **LLM Router** (with reasoning tags), dynamically directing questions to the optimal translation technique (HyDE, Step-Back, Decomposition, RAG-Fusion, Multi-Query, or Standard retrieval).
* 🏷️ **Metadata Filtering & Query Analysis**: Extracts hard filters (e.g. `publish_year`, `file_type`, `page_number`, and `data_source` corpus categories) using a structured Pydantic Query Analyzer, matching them against automatically enriched metadata. Resolves relative temporal expressions (e.g. "last year") dynamically by injecting the system clock.
* 🕸️ **Stateful LangGraph Agent**: Uses a cyclic state-graph workflow for multi-step or comparative query decomposition, handling sub-tasks sequentially with contextual memory tracking.
* 📌 **Multi-Representation Indexing**: At ingestion time, an LLM generates a clean one-line summary for every chunk (in parallel). The vector database indexes the clean summary for accurate semantic matching, but retrieval transparently returns the full original chunk content to the answer generator.
* 🌲 **RAPTOR Tree Summaries** *(opt-in via `--raptor` flag)*: Recursively clusters document chunks and intermediate summaries using Gaussian Mixture Models (GMM) to build a multi-level tree of information (from detailed leaves to high-level root summaries), embedding all levels to answer broad, global queries.
* 🤖 **Agentic Self-Reflective RAG (CRAG + Self-RAG)**: For complex analytical queries (compare, evaluate, contrast, etc.), the pipeline activates a LangGraph CRAG loop that grades retrieved documents for relevance, rewrites the query if needed, generates an answer, and self-reflects to detect and correct hallucinations before returning the response.
* 🧠 **Semantic Chunking**: Instead of static character-limit splits, the pipeline calculates similarity drift between consecutive sentences to keep related concepts together.
* 🔍 **Hybrid Query Matching**: Fuses vector similarity (dense search) with term-frequency index scanning (BM25 sparse search) to capture both concepts and exact keywords.
* ⚖️ **HyDE Hallucination Guardrail**: Performs a cosine similarity embedding check on the generated hypothetical passage, falling back to standard retrieval if similarity drops below `0.60`.
* ⚡ **Parallel Retrieval Execution**: Runs sub-queries concurrently via a Python ThreadPool executor to ensure near-zero latency overhead for multi-query strategies.
* 🎯 **Configurable Reranking**: Supports either local, zero-cost quantized Cross-Encoder scoring via `Flashrank`, or cloud-based enterprise `Cohere Rerank` integration, selectable in the environment configuration.
* 💬 **History-Aware Contextualization**: Translates conversational pronouns (e.g. *"What is task decomposition?"* $\rightarrow$ *"Give me an example of it"*) into self-contained search terms.
* 📚 **Fact-Checking Citations**: Every generated answer is paired with the exact source title, source URL, and a snippet preview.

---

## 📈 Architecture & Diagrams

### 1. Data Ingestion Pipeline
```mermaid
flowchart TD
    Start([Start Ingestion]) --> DirCheck{Are there files in ./documents/?}
    
    %% Web Fallback Route
    DirCheck -->|No| WebLoad[WebBaseLoader: Scraping Lilian Weng Blog]
    WebLoad --> Parse[BeautifulSoup SoupStrainer: Filter tags]
    
    %% Directory Route
    DirCheck -->|Yes| ScanDir[Scan ./documents/*]
    ScanDir --> Router{File Extension?}
    Router -->|.pdf| PDF[PyPDFLoader]
    Router -->|.csv| CSV[CSVLoader]
    Router -->|.docx| Word[Docx2txtLoader]
    Router -->|.txt / .md| Text[TextLoader]
    
    %% Accumulation & Splitting
    PDF & CSV & Word & Text --> Accumulate[Accumulate Documents]
    Parse --> Accumulate
    
    Accumulate --> EmbeddingInit[OpenAIEmbeddings: text-embedding-3-small]
    EmbeddingInit --> Chunking[SemanticChunker: Percentile Thresholds]
    Chunking --> Splits[Generate Semantic Chunks]
    
    %% Persistence
    Splits --> VectorStoreFAISS[(Save to Local ./faiss_db)]
    VectorStoreFAISS --> End([Ingestion Complete])
    
    style VectorStoreFAISS fill:#ff9900,stroke:#333,stroke-width:2px
    style Chunking fill:#2ecc71,stroke:#333,stroke-width:2px
    style Router fill:#3498db,stroke:#333,stroke-width:2px
```

### 2. Multi-Query Hybrid Retrieval Pipeline
```mermaid
flowchart TD
    Input([User Query]) --> ContextChecker[main.py: Validate Env & DB]
    ContextChecker --> MultiQuery[MultiQueryRetriever: Generate 3 Variations]
    
    %% Parallel Retrieval Loop
    subgraph Parallel Retrieval per Query Variation
        MultiQuery --> BM25[BM25Retriever: Sparse Keyword Match]
        MultiQuery --> Vector[FAISSRetriever: Dense Vector Similarity]
    end
    
    BM25 -->|Retrieve top-8| Merger[EnsembleRetriever: Reciprocal Rank Fusion]
    Vector -->|Retrieve top-8| Merger
    
    %% Re-ranking
    Merger -->|Fused Candidate List| Reranker[FlashrankRerank: Cross-Encoder]
    Reranker -->|Re-scored & Sorted| TopK[Select Top-3 Documents]
    
    %% Prompt Generation
    TopK --> LLM[ChatOpenAI: gpt-4o-mini]
    LLM --> Answer([Format Output Answer + Source Citations])
    
    style Vector fill:#5dade2,stroke:#333,stroke-width:1px
    style Reranker fill:#f1c40f,stroke:#333,stroke-width:2px
    style LLM fill:#e74c3c,stroke:#333,stroke-width:2px
```

### 3. Request Lifecycle & Conversational Memory
```mermaid
sequenceDiagram
    autonumber
    actor User as Developer/CLI
    participant Main as main.py (Runtime)
    participant Hist as Chat History State
    participant LLM as OpenAI Chat API (gpt-4o-mini)
    participant Retriever as Multi-Query Hybrid Retriever

    User->>Main: Launch application
    Main->>Hist: Initialize chat_history = []
    
    loop Conversation Loop
        User->>Main: Ask question (e.g. "What is task decomposition?")
        Main->>Retriever: Query with current input & chat_history
        Note over Retriever: Reformulates query contextually to standalone question.
        Retriever-->>Main: Return top-3 re-ranked source documents
        Main->>LLM: Send Context (Docs) + Chat History + Input
        LLM-->>Main: Return Generated Answer
        Main->>User: Display Answer + Document Source Citations
        
        %% Update memory
        Main->>Hist: Append HumanMessage(input)
        Main->>Hist: Append AIMessage(answer)
    end
```

---

## 📊 Baseline RAG vs. Advanced RAG

| Optimization Stage | Baseline RAG | Advanced RAG (This Project) | Impact |
| :--- | :--- | :--- | :--- |
| **Splitting** | Character Count (Fixed) | Semantic Similarity Splitter | Preserves thematic sentences together |
| **Retrieval** | Semantic Search only | Hybrid Search (Vector + BM25 Keyword) | Doesn't miss exact names/part numbers |
| **Translation** | Standard Query | Multi-Query Variation Generation | Resolves poorly phrased user questions |
| **Ordering** | Simple Vector Distance | Local Cross-Encoder Re-scoring | Eliminates context window noise |
| **Memory** | None (Single Turn) | Stateful Conversational History | Understands context of follow-ups |

---

## 🛠️ Tech Stack

* **Package Manager**: [uv](https://github.com/astral-sh/uv) (Rust-powered, ultra-fast Python environment sync)
* **LLM Engine**: OpenAI API (`gpt-4o-mini` & `text-embedding-3-small`)
* **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search - lightweight, local, network-vulnerability-free vector library)
* **Sparse Index**: [Rank-BM25](https://github.com/dorianbrown/rank_bm25)
* **Re-ranker Model**: [Flashrank](https://github.com/prithivida/flashrank) (runs locally using ONNX, zero API keys required)
* **Agent Framework**: [LangGraph](https://github.com/langchain-ai/langgraph) (for cyclic decomposition loops)
* **Document Processing**: `pypdf`, `docx2txt`, `beautifulsoup4`, `tiktoken`

---

## 📂 Project Layout

```plaintext
rag_project/
│
├── .env.example        # Reference configurations (Template)
├── .gitignore          # Safeguards to prevent committing .env and faiss_db
├── pyproject.toml      # Modern PEP 518/621 project configuration managed by uv
├── requirements.txt    # Shared dependency list
│
├── ingest.py           # Scraping, parsing, chunking, Multi-Rep + RAPTOR embedding pipeline
├── query_processor.py  # Dynamic routing engine with 5 query translation techniques
├── decomposition_graph.py # Stateful LangGraph agent for cyclic decomposed queries
├── agentic_graph.py    # CRAG + Self-RAG LangGraph agent for complex analytical queries
├── multi_rep_utils.py  # Multi-Representation Indexing: summary generation & content restore
├── main.py             # User interface, database retriever load, and generation logic
├── playground.py       # Helper playground for similarity calculations and token sizes
│
├── documents/          # Directory where local PDFs, CSVs, and Word files are placed
└── faiss_db/           # Persistent local directory containing vector indices (Git ignored)
```

---

## 🚀 Getting Started

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sh1vaay/RAG.git
   cd RAG
   ```

2. **Synchronize dependencies:**
   `uv` automatically configures your virtual environment and locks dependencies:
   ```bash
   uv sync
   ```

### Configuration (`.env`)

Copy the example configuration file:
```bash
cp .env.example .env
```
Open `.env` and configure your settings:
```ini
OPENAI_API_KEY=sk-proj-YOUR_API_KEY

# Routing configuration: 'semantic' (Fast/Free similarity matching) or 'llm' (Structured reasoning)
ROUTING_METHOD=semantic
```

### Usage

1. **Populate Documents**: Place your PDFs, Word documents (`.docx`), CSVs, or text files into the `./documents` folder.
2. **Ingest Data**: Execute the parser and chunking pipeline to build the database:
   ```bash
   uv run ingest.py
   ```
3. **Launch the Chat CLI**: Run the interactive conversational loop:
   ```bash
   uv run main.py
   ```

---

## 🔒 Security & Optimization

* **Secrets Management**: Built-in protections inside `.gitignore` ensure `.env` and local database binary caches (`faiss_db/`) are blocked from version control.
* **Token Boundaries**: Semantic chunking breaks text without exceeding context-window limits, preventing model truncation and context cost bloat.
* **Fast Startup**: Hybrid search indexes are serialized and loaded from disk locally, bypassing repeated document scraping.

---

## 📈 Monitoring & Observability

This project includes integrations with **LangSmith** to inspect prompts, retrieval steps, latencies, and token costs:

```ini
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__YOUR_LANGSMITH_API_KEY
LANGCHAIN_PROJECT="rag-local-assistant"
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
