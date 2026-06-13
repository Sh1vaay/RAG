# 🌌 Aether AI: Enterprise Corrective & Self-Reflective RAG Engine

<div align="center">

*A production-grade, locally-persisted Corrective & Self-Reflective Conversational RAG pipeline built on high-fidelity query optimizations, dynamic routing, and multi-agent LangGraph workflows.*

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
* [⚡ Key Features](#-key-features)
* [🏗️ System Architecture](#️-system-architecture)
* [🔄 Application Request Lifecycle Flow](#-application-request-lifecycle-flow)
* [🔀 Query Routing Decision Flow](#-query-routing-decision-flow)
* [🛠️ Technology Stack](#️-technology-stack)
* [🔧 Design Decisions](#-design-decisions)
* [🔌 API Reference Specs](#-api-reference-specs)
* [📁 Folder Structure](#-folder-structure)
* [🚀 Developer Experience & Setup](#-developer-experience--setup)
  * [Environment Configuration](#environment-configuration)
  * [Installation Steps](#installation-steps)
  * [CLI Execution](#cli-execution)
  * [FastAPI Server Execution](#fastapi-server-execution)
* [🐳 Production Deployment (Docker)](#-production-deployment-docker)
* [🔒 Security & Hardening Policy](#-security--hardening-policy)
* [📈 Performance, Scalability & Observability](#-performance-scalability--observability)
* [🤝 Contributing Guidelines](#-contributing-guidelines)
* [📄 License](#-license)

---

## 🪐 Project Overview

Standard Retrieval-Augmented Generation (RAG) systems frequently struggle in production due to three core challenges: **retrieval noise** (injecting irrelevant text), **hallucinations** (unsupported model outputs), and **high latency** (processing simple tasks through heavy pipelines).

**Aether AI** resolves these issues by organizing RAG execution into a **Dual-Path processing topology**:
1. **The Fast Path (Low-Latency Bypass)**: Simple inputs, greetings, or direct conversation bypass heavy vector stores entirely using an in-memory embedding-based **Semantic Router**, routing queries to a lightweight conversational agent in milliseconds.
2. **The Heavy Path (Self-Reflective Agents)**: Complex queries are routed to specialized pipelines where Pydantic Query Analyzers parse constraints (years, pages, types). Retrieval combines dense (FAISS) and sparse (BM25) search indices via Reciprocal Rank Fusion (RRF), followed by a multi-agent **LangGraph** self-correcting loop. If retrieved documents are irrelevant, the system triggers **Corrective RAG (CRAG)** via DuckDuckGo Web Search. Final answers are validated against hallucinations before reaching the user.

---

## ⚡ Key Features

* 🔮 **Dual-Routing Switchboard**: Supports embedding-based local **Semantic Routing** (zero-token latency, in-memory) and **LLM Routing** (gpt-4o-mini structured schema) to classify query intent.
* 🔬 **Pydantic Query Analyzer**: Automatically extracts database metadata filters (e.g., `publish_year`, `file_type`, `page_number`) and resolves relative dates (e.g., "last year") to build strict database queries.
* 🔀 **Hybrid Retrieval (RRF)**: Combines dense vector retrieval (FAISS) and sparse keyword retrieval (BM25) using Reciprocal Rank Fusion to ensure both semantic capture and exact keyword matches.
* 🕸️ **LangGraph Multi-Hop Decomposition**: Breaks complex, multi-faceted questions into sequential sub-questions, answering them one-by-one using intermediate context memory.
* 🌐 **Corrective RAG (CRAG)**: Grades retrieved documents and dynamically triggers DuckDuckGo web search to gather missing facts when local context is insufficient.
* 🪞 **Double-Guardrail Self-RAG Evaluator**: Uses a two-step validation chain (Hallucination Grader + Answer Relevance Grader) to run verification loops and query rewrites until the output is fully grounded.
* 🌲 **Hierarchical RAPTOR Indexing**: Builds a multi-level tree of document chunks and cluster summaries using Gaussian Mixture Models (GMM) to support high-fidelity global summarization.
* ✂️ **Semantic Chunking**: Identifies meaning-based boundaries by tracking semantic drift across adjacent sentences, preventing paragraph truncation.
* 🎯 **Flashrank CPU Reranking**: Re-ranks candidates locally on CPU using optimized quantized cross-encoder models.

---

## 🏗️ System Architecture

The following diagram illustrates Aether AI's multi-layered layout, showing data flows from document ingestion down to response validation:

```mermaid
graph TB
    subgraph UI ["🖥️ Presentation Layer (Frontend / Client)"]
        Browser["🌐 Web Browser (HTML5/JS Dashboard)"]
        CLI["💻 CLI Client (src/main.py)"]
    end

    subgraph API ["⚡ API & Orchestration Layer (Backend)"]
        FastAPI["🚀 FastAPI App (src/app.py)"]
        Router["🔀 Query Switchboard (src/query_processor.py)"]
        LangGraph["🕸️ LangGraph Agents (src/agentic_graph.py / src/decomposition_graph.py)"]
    end

    subgraph DATA ["💾 Data & Storage Layer"]
        FAISS["🧲 FAISS Vector DB"]
        BM25["📄 BM25 Sparse Index"]
        FS["📁 File System (staged PDFs/DOCX)"]
    end

    subgraph EXT ["🌐 External Services"]
        OpenAI["🤖 OpenAI API (gpt-4o-mini & embeddings)"]
        Cohere["☁️ Cohere Rerank API (Optional)"]
        DDG["🌍 DuckDuckGo (CRAG Web Search)"]
        LangSmith["📊 LangSmith (Tracing & Observability)"]
    end

    Browser <-->|HTTP / JSON| FastAPI
    CLI <-->|Local Python Calls| Router
    FastAPI --> Router
    Router -->|Orchestrates| LangGraph
    
    %% Data retrieval
    LangGraph -->|Read Vectors| FAISS
    LangGraph -->|Read Keyword Index| BM25
    FastAPI -->|Load Documents| FS
    
    %% External API calls
    LangGraph -->|Embeddings / Gen| OpenAI
    Router -->|Embeddings / Gen| OpenAI
    LangGraph -->|Rerank Context| Cohere
    LangGraph -->|Web Search| DDG
    FastAPI -->|Telemetry| LangSmith
    LangGraph -->|Telemetry| LangSmith
    Router -->|Telemetry| LangSmith
```

---

## 🔄 Application Request Lifecycle Flow

This sequence flowchart shows how user requests are parsed, routed, retrieved, graded, and validated:

```mermaid
flowchart TD
    Start(["🧑‍💻 User Query"]) --> Contextualize{"📜 Has Chat History?"}
    Contextualize -->|"Yes"| RewriteQ["🤖 GPT-4o-mini<br/>(Generate Standalone Question)"]
    Contextualize -->|"No"| UseOriginal["Use Original Question"]
    
    RewriteQ --> RouteStep
    UseOriginal --> RouteStep
    
    RouteStep["🔮 Router Classifies Query"] --> RouteDecision{"Select Route"}
    
    %% Fast Path
    RouteDecision -->|"⚡ simple"| FastPath["🚀 Fast Path Bypass"]
    FastPath --> FastRetrieve["Basic Retrieve k=3"]
    FastRetrieve --> FastGen["🤖 GPT-4o-mini Q&A"]
    FastGen --> EndResponse
    
    %% Heavy Path
    RouteDecision -->|"🧠 complex / multi-hop"| Analyzer["🔬 Pydantic Query Analyzer"]
    Analyzer --> ExtractedFilters["🎯 Extracted Metadata Filters"]
    ExtractedFilters --> HybridSearch["🔀 Hybrid Retrieve: FAISS + BM25 + RRF"]
    
    HybridSearch --> DecisionWorkflow{"Route Type?"}
    
    %% Decomposition
    DecisionWorkflow -->|"decomposition"| GraphDec["🕸️ LangGraph Multi-Hop Agent"]
    GraphDec --> SubQ["Decompose Sub-Questions"]
    SubQ --> SequentialAns["Answer Sequentially with Context Memory"]
    SequentialAns --> Synthesis["Synthesize Final Answer"]
    
    %% Standard / CRAG / Self-RAG
    DecisionWorkflow -->|"standard / fusion / step_back / hyde"| GraphCRAG["🕸️ LangGraph Self-Correcting Agent"]
    GraphCRAG --> Grader["⚖️ Grade Retrieved Documents"]
    Grader -->|"❌ Irrelevant"| Rewriter["✏️ Query Rewriter"]
    Rewriter --> WebSearch["🌐 DuckDuckGo Web Search"]
    WebSearch --> GenResponse
    
    Grader -->|"✅ Relevant"| GenResponse["💡 Generate Grounded Response"]
    
    GenResponse --> SelfReflect{"🪞 Hallucination Grader"}
    SelfReflect -->|"⚠️ Hallucinated"| Rewriter
    SelfReflect -->|"✅ Grounded"| AnswerRelevance{"✅ Answer Grader"}
    AnswerRelevance -->|"❌ Off-Topic"| Rewriter
    AnswerRelevance -->|"✅ Addresses Q"| Synthesis
    
    Synthesis --> EndResponse(["🎯 Final Response + Citations"])
```

---

## 🔀 Query Routing Decision Flow

The diagram below details the step-by-step decision routing logic executed within the `RoutingRetriever` (both Semantic and LLM-based) and the main runtime orchestrator:

```mermaid
flowchart TD
    Start(["🧑‍💻 User Query"]) --> ContextCheck{"📜 Has Chat History?"}
    ContextCheck -->|"Yes"| Rewrite["🤖 GPT-4o-mini Q-Rewriter<br/>(Generate Standalone Question)"]
    ContextCheck -->|"No"| Standalone["Use Original Question"]
    
    Rewrite --> RouteEngine
    Standalone --> RouteEngine
    
    subgraph RouteEngine ["🔮 Routing Switchboard (RoutingRetriever)"]
        DetermineMethod{"Check ROUTING_METHOD"}
        
        DetermineMethod -->|"llm"| LLMRoute["🤖 GPT-4o-mini Structured Routing<br/>(RouteSelection schema)"]
        DetermineMethod -->|"semantic (default)"| SemRoute["⚡ Zero-Dependency Semantic Router<br/>(Cosine Similarity vs Reference Samples)"]
        
        LLMRoute --> RouteCheck
        SemRoute --> SimilarityCheck{"Max Similarity >= 0.40?"}
        SimilarityCheck -->|"Yes"| RouteCheck{"Select Route"}
        SimilarityCheck -->|"No"| StandardFallback["standard"]
        
        RouteCheck -->|"simple"| RouteSimple["simple"]
        RouteCheck -->|"hyde"| RouteHyDE["hyde"]
        RouteCheck -->|"step_back"| RouteSB["step_back"]
        RouteCheck -->|"decomposition"| RouteDec["decomposition"]
        RouteCheck -->|"rag_fusion"| RouteFusion["rag_fusion"]
        RouteCheck -->|"multi_query"| RouteMQ["multi_query"]
        RouteCheck -->|"standard"| RouteStd["standard"]
        StandardFallback --> RouteStd
    end

    RouteSimple -->|"⚡ Bypasses Heavy Pipeline"| FastPath
    RouteHyDE --> HeavyPath
    RouteSB --> HeavyPath
    RouteDec --> HeavyPath
    RouteFusion --> HeavyPath
    RouteMQ --> HeavyPath
    RouteStd --> HeavyPath

    subgraph FastPath ["⚡ Fast Path (Low-Latency Bypass)"]
        ExecuteFast["Retrieve Basic (k=3)"] --> GenFast["🤖 Q&A (gpt-4o-mini)"]
    end

    subgraph HeavyPath ["🧠 Heavy Path Workflows"]
        QA["🔬 Pydantic Query Analyzer"] --> ExtractFilters["Extract Metadata Filters<br/>(file_type, year, page, source)"]
        ExtractFilters --> ApplyFilters["Apply Hard Filters to FAISS Vector DB"]
        
        ApplyFilters --> RouteDispatcher{"Route Check & Context Match?"}
        
        %% Decomposition Graph
        RouteDispatcher -->|"decomposition"| GraphDec["🕸️ LangGraph Decomposition Agent"]
        GraphDec --> SubQ["Decompose into sequential sub-questions"]
        SubQ --> SolveSub["Answer sequentially with memory"]
        SolveSub --> SynthesizeDec["Synthesize Final Response"]
        
        %% Self-RAG / CRAG Graph
        RouteDispatcher -->|"standard + comparative keywords"| GraphAgentic["🕸️ LangGraph Self-Correcting Agent"]
        GraphAgentic --> GradeDocs{"Grade Retrieved Documents"}
        GradeDocs -->|"❌ Irrelevant"| CRAGWeb["🌐 DuckDuckGo Web Search Fallback"]
        GradeDocs -->|"✅ Relevant"| GenGrounded["Generate Grounded Answer"]
        CRAGWeb --> GenGrounded
        GenGrounded --> SelfRAGGrade{"🪞 Hallucination & Relevance Grader"}
        SelfRAGGrade -->|"❌ Failed"| Rewriter["✏️ Query Rewriter"] --> CRAGWeb
        SelfRAGGrade -->|"✅ Passed"| SynthesizeAgentic["Compile Final Response"]
        
        %% Retriever Translation paths
        RouteDispatcher -->|"hyde / step_back / rag_fusion / multi_query / standard"| RunRetrievalStrategy{"Retrieve with Strategy"}
        
        RunRetrievalStrategy -->|"hyde"| HyDECheck{"HyDE Similarity >= 0.60?"}
        HyDECheck -->|"Yes"| RetrieveHyDE["Retrieve using hypothetical passage"]
        HyDECheck -->|"No"| RetrieveOrig["Retrieve using original question"]
        
        RunRetrievalStrategy -->|"step_back"| RetrieveSBQuery["Retrieve for original + abstract concept queries"]
        
        RunRetrievalStrategy -->|"rag_fusion"| RetrieveRRF["Retrieve multi-perspectives in parallel + RRF (k=60, top_n=8)"]
        
        RunRetrievalStrategy -->|"multi_query"| RetrieveMQQueries["Retrieve multi-perspectives in parallel & merge"]
        
        RunRetrievalStrategy -->|"standard"| RetrieveDirect["Retrieve directly using Compression Retriever"]
        
        RetrieveHyDE & RetrieveOrig & RetrieveSBQuery & RetrieveRRF & RetrieveMQQueries & RetrieveDirect --> PostFilter["Apply Post-Filtering on Metadata"]
        PostFilter --> RestoreContent["Restore original content (swap back Multi-Rep summaries)"]
        RestoreContent --> GenStandard["🤖 GPT-4o-mini Q&A Chain"]
    end

    GenFast --> EndResponse(["🎯 Output Response + Citations"])
    SynthesizeDec --> EndResponse
    SynthesizeAgentic --> EndResponse
    GenStandard --> EndResponse

    style FastPath fill:#f0f9ff,stroke:#0284c7,stroke-dasharray: 5 5,stroke-width:2px
    style HeavyPath fill:#faf5ff,stroke:#7e22ce,stroke-dasharray: 5 5,stroke-width:2px
    style RouteEngine fill:#f8fafc,stroke:#475569,stroke-dasharray: 5 5,stroke-width:2px
```

---

## 🛠️ Technology Stack

* **Core Language & Tooling**: Python 3.12+ managed by [uv](https://github.com/astral-sh/uv) (Rust-powered virtual environment manager).
* **Agentic Framework**: [LangGraph v0.2](https://github.com/langchain-ai/langgraph) & [LangChain v1.0 / langchain-core](https://github.com/langchain-ai/langchain).
* **Vector Store**: [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss) CPU-optimized local database.
* **Sparse Search Engine**: [Rank-BM25](https://github.com/dorianbrown/rank_bm25).
* **Reranker Model**: [Flashrank](https://github.com/prithivida/flashrank) running local quantized cross-encoders via ONNX Runtime.
* **API Framework**: [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) ASGI server.
* **Frontend**: Vanilla HTML5/JS single-page dashboard styled with Tailwind CSS (Glassmorphism theme).
* **Document Parsing**: `pypdf`, `docx2txt`, `beautifulsoup4`, `csv`.

---

## 🔧 Design Decisions

1. **Local Vector Storage (FAISS)**: Using local FAISS database directory indexes keeps operational costs low, eliminates cloud dependencies, and ensures data remains stored locally.
2. **Dense-Sparse Hybrid Retrieval**: Dense embeddings model high-level concepts but fail on precise identifiers like codes, years, or numbers. Combining FAISS with BM25 via Reciprocal Rank Fusion (RRF) preserves both search qualities.
3. **Decoupled Evaluators**: Evaluation agents in LangGraph are configured as isolated nodes with temperature=0 to prevent stochastic variance during verification.
4. **Asynchronous Execution**: Ingestion operations run in separate system-level processes using `asyncio` to prevent blocking the main server thread.

---

## 🔌 API Reference Specs

### Endpoints Overview

| Method | Endpoint | Description | Request Payload | Response Schema |
| :--- | :--- | :--- | :--- | :--- |
| **GET** | `/` | Serves the main UI Dashboard page | None | `text/html` |
| **GET** | `/api/status` | Returns DB metrics, config, and staged files | None | JSON status object |
| **POST** | `/api/config` | Updates routing and reranker configurations | `ConfigUpdateRequest` | Success message |
| **POST** | `/api/chat` | Evaluates query and executes RAG pipeline | `ChatRequest` | `ChatResponse` |
| **POST** | `/api/upload` | Uploads raw documents safely to `./documents/` | Multipart Files | List of saved filenames |
| **POST** | `/api/ingest` | Parses staged files and builds vector DB | Query: `raptor` (bool) | Ingestion log snippet |

---

## 📂 Folder Structure

```plaintext
rag_project/
├── .github/workflows/
│   └── ci.yml              # CI workflow checking syntax, lint, & Ruff formatting
├── documents/              # Staging area for raw source documents (PDF, CSV, MD, etc.)
├── faiss_db/               # Locally-persisted FAISS vector index files
├── .dockerignore           # Excludes local caches, database files, and secrets from builds
├── .env.example            # Configuration settings template
├── .gitignore              # Ignores local databases, virtual envs, and API credentials
├── Dockerfile              # Secure multi-stage production Dockerfile
├── pyproject.toml          # Project dependencies, linter settings, and metadata (uv-managed)
├── requirements.txt        # Package locking file for pinning versions
│
└── src/
    ├── app.py              # FastAPI server and application endpoints
    ├── main.py             # Core setup logic and CLI execution loop
    ├── ingest.py           # Ingest pipeline (semantic splitting, GMM/RAPTOR index creation)
    ├── query_processor.py  # Embedding classifier and Pydantic SearchQuery analyzer
    ├── agentic_graph.py    # LangGraph CRAG and Self-RAG state-graph
    ├── decomposition_graph.py # LangGraph multi-hop sequential decomposition agent
    └── multi_rep_utils.py  # Multi-Representation Indexing utilities
```

---

## 🚀 Developer Experience & Setup

### Environment Configuration

1. Copy the configuration template:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and fill in your keys:
   ```ini
   OPENAI_API_KEY=sk-proj-YOUR_API_KEY_HERE
   
   # Optional LangSmith tracing config
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=lsv2_pt_...
   LANGCHAIN_PROJECT="rag-telemetry-dashboard"
   ```

### Installation Steps

1. **Install uv** (Rust-powered package manager):
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Install project dependencies**:
   ```bash
   uv sync
   ```

### Ingesting Documents

Place your files in `./documents/`, then run the parser:
```bash
# Standard Ingestion & Semantic Chunking
uv run python -m src.ingest

# Advanced Ingestion (RAPTOR clustering tree enabled)
uv run python -m src.ingest --raptor
```

### CLI Execution

To run queries directly in your terminal:
```bash
uv run python -m src.main
```

### FastAPI Server Execution

To run the FastAPI server locally:
```bash
uv run python -m src.app
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser to access the dashboard.

---

## 🐳 Production Deployment (Docker)

This project uses a secure multi-stage Docker build to build lightweight runtime containers.

1. **Build the Container Image**:
   ```bash
   docker build -t aether-rag-service .
   ```
2. **Launch the Container**:
   Pass your keys and mount local folders to persist the database files:
   ```bash
   docker run -d \
     -p 8000:8000 \
     -e OPENAI_API_KEY="sk-proj-YOUR_KEY" \
     -v $(pwd)/documents:/app/documents \
     -v $(pwd)/faiss_db:/app/faiss_db \
     aether-rag-service
   ```

---

## 🔒 Security & Hardening Policy

* **Dependencies Hardening**: Vulnerable packages prone to arbitrary code execution during deserialization or pickling have been removed from the dependency tree.
* **Dynamic Config Verification**: Dynamic configs via `/api/config` are validated against schemas before being applied, protecting the backend from runtime injection.
* **Safe Subprocesses**: Subprocess execution in `/api/ingest` uses explicit arguments list parsing instead of shell integration (`shell=False`), preventing command injections.
* **Data Path Traversal Protections (CWE-22 / CWE-23)**: In `/api/upload`, uploaded filenames are stripped of path components like `../` and null bytes (`\0`) to keep files contained inside `./documents/`.
* **Safe Error Logging**: FastAPI endpoints are configured to catch internal exceptions, logging full tracebacks to `sys.stderr` while returning clean, generic messages to the browser.
* **Non-Root Execution (Principle of Least Privilege)**: The production Docker container creates a user `appuser` (UID 10001) and runs the FastAPI server under this user.

---

## 📈 Performance, Scalability & Observability

* **Parallel Searches**: Multi-query expansions are run concurrently using `ThreadPoolExecutor` to minimize LLM overhead.
* **Quantized Reranking**: Context reranking is executed on CPU inside local ONNX runtimes using Flashrank, bypassing network calls to external reranking APIs.
* **Memory Management**: Vector index and database lookups are loaded on demand and cached inside memory.
* **LangSmith Tracing**: Full runtime tracing is configured. Turning on `LANGCHAIN_TRACING_V2` in `.env` automatically captures step execution latencies, input/output logs, and costs.

---

## 🤝 Contributing Guidelines

1. **Formatting**: Format all edits using Ruff before staging:
   ```bash
   uv run ruff format .
   ```
2. **Syntax Validation**: Ensure that the project compiles cleanly:
   ```bash
   uv run python -m compileall .
   ```
3. **Open a PR**: Send pull requests to the `main` branch with descriptive change logs.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more details.
