# 🌌 Advanced Local Conversational RAG System

<div align="center">

*An enterprise-ready, locally persisted Corrective & Self-Reflective Conversational RAG pipeline built on high-fidelity query optimizations and multi-agent LangGraph workflows.*

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
* [🏗️ System Architecture](#️-system-architecture)
* [🔄 Application Request Flow](#-application-request-flow)
* [🛠️ Tech Stack](#️-tech-stack)
* [📂 Project Layout](#-project-layout)
* [🚀 Getting Started](#-getting-started)
  * [Configuration (`.env`)](#configuration-env)
  * [Installation](#installation)
  * [Usage](#usage)
* [🔒 Security & Hardening](#-security--hardening)
* [📈 Performance & Scalability](#-performance--scalability)
* [📄 License](#-license)

---

## 🪐 Project Overview

Standard RAG architectures frequently suffer from retrieval noise, model hallucinations, and high latency when processing mixed datasets. 

This project implements an **Advanced Offline Conversational RAG Pipeline** designed for high recall, precision, and safety. It features:
* **Upfront Routing**: Instantly separates simple conversational tasks (Fast Path) from heavy multi-hop retrieval paths using an in-memory Semantic Router.
* **Robust Multi-Query Retrieval**: Executes hybrid dense (FAISS) and sparse (BM25) searches, and combines candidate documents using Reciprocal Rank Fusion (RRF).
* **Multi-Agent Orchestration**: Uses LangGraph workflows to decompose complex questions into sequential sub-tasks with memory tracking.
* **Double-Guardrail Self-RAG Loop**: Validates the correctness and usefulness of generated answers using an LLM evaluator to prevent hallucinations.
* **Corrective RAG (CRAG)**: Dynamically falls back to Google/DuckDuckGo web search if local document indexing lacks relevant facts.

---

## ⚡ Key Capabilities

* 📂 **Multi-Format Processing**: Routes PDFs (`PyPDFLoader`), CSVs (`CSVLoader`), Word files (`Docx2txtLoader`), and text documents directly from a local `./documents` folder.
* 🔮 **Configurable Dual-Routing Engine**: Features an in-memory embedding-based **Semantic Router** (zero-token latency) and an **LLM Router** (gpt-4o-mini structured output) to select optimal query translation strategies (HyDE, Step-Back, Decomposition, RAG-Fusion, Multi-Query, or Simple bypass).
* 🏷️ **Metadata Filtering & Temporal Analysis**: Dynamically extracts constraints like `publish_year`, `file_type`, `page_number`, and `data_source` corpus categories using Pydantic Query Analyzers. Relative time mentions (e.g., "last year") are resolved dynamically against the system clock.
* 🕸️ **LangGraph Query Decomposition**: Sequentially processes multi-hop questions using a cyclic state-graph workflow, updating sub-answers and search context dynamically.
* 🤖 **Corrective RAG (CRAG) Fallback**: Integrates LangChain's `DuckDuckGoSearchRun` to augment context with real-time web facts when retrieved local documents are evaluated as irrelevant.
* 🪞 **Double-Guardrail Evaluator (Self-RAG)**: 
  1. **Hallucination Grader**: Checks if the generated answer is fully grounded in the retrieved facts.
  2. **Answer Grader**: Validates if the answer actually addresses the user's original query. If validation fails, the loop triggers a search query rewrite and web search fallback.
* 📌 **Multi-Representation Indexing**: Summarizes raw chunks into clean one-line summaries at ingestion. The database indexes these summaries for optimal semantic alignment, but retrieval swaps them back to their original chunks to preserve rich context for generation.
* 🌲 **RAPTOR Tree Summaries** *(opt-in via `--raptor` flag)*: Recursively clusters document chunks and intermediate summaries using Gaussian Mixture Models (GMM) to build a multi-level tree of information (from detailed leaves to high-level root summaries), embedding all levels to answer broad, global queries.
* 🧠 **Semantic Chunking**: Computes semantic similarity drift between consecutive sentences to keep related concepts grouped together in cohesive chunks.
* 🔍 **Hybrid Query Matching**: Fuses dense similarity matching (FAISS) with term-frequency index scanning (BM25) to capture both semantic concepts and exact keywords.
* 🎯 **Configurable Reranking**: Supports local Cross-Encoder scoring via `Flashrank`, or cloud-based `Cohere Rerank` integration, selectable in the environment configuration.

---

## 🏗️ System Architecture

The following diagram illustrates the components, indices, services, and execution logic of the system:

```mermaid
flowchart TB
    %% Ingestion Section
    subgraph Ingestion [1. Ingestion Pipeline]
        Docs[("./documents/ (PDF, CSV, Docx, TXT)")] --> Loaders["Document Loaders (PyPDFLoader, CSVLoader, etc.)"]
        Loaders --> Chunking["Semantic Chunking (Percentile Drift)"]
        Chunking --> Splits["Raw Document Splits"]
        Splits --> MultiRep["Multi-Representation Summary Generation"]
        Splits -.->|--raptor| GMM["RAPTOR: Gaussian Mixture Model Clustering"]
        GMM -.-> Tree["Multi-Level Hierarchical Tree Summaries"]
        MultiRep & Tree --> StoreFAISS[("Local FAISS Vector DB\n(index: text-embedding-3-small)")]
    end

    %% Routing & Processing Section
    subgraph Execution [2. Query Execution & Routing]
        UserQuery([User Input Query]) --> HistoryChecker["Chat History Contextualization"]
        HistoryChecker --> Router{"Semantic / LLM Router"}
        
        Router -->|simple| FastPath["⚡ Fast Path (Simple/Greeting Bypass)"]
        FastPath --> FastGen["Lightweight Generation Chain"]
        
        Router -->|complex| Analyzer["Pydantic Query Analyzer (Metadata Filters)"]
        Analyzer --> FilterInject["FAISS Vector Store Dynamic Filtering"]
    end

    %% LangGraph Agents
    subgraph HeavyPipeline [3. Deep Retrieval & Self-RAG Agents]
        FilterInject --> StandardRet["Multi-Query Hybrid Retriever\n(FAISS Dense + BM25 Sparse + RRF)"]
        
        FilterInject -->|decomposition route| DecGraph["LangGraph: Sequential Multi-Hop Agent"]
        DecGraph --> Synthesize["Synthsized Sub-answers"]
        
        StandardRet --> GradeDocs{"Document Relevance Grader"}
        GradeDocs -->|Irrelevant Docs| RewriteQuery["Query Rewriter Node"]
        RewriteQuery --> DDGWeb["DuckDuckGo Web Search Fallback"]
        DDGWeb --> GenerateNode["Generate Answer Node"]
        
        GradeDocs -->|Relevant Docs| GenerateNode
        Synthesize --> GenerateNode
        
        GenerateNode --> HallucinationGrader{"Hallucination Grader\n(Is grounded?)"}
        HallucinationGrader -->|No - Hallucinated| RewriteQuery
        HallucinationGrader -->|Yes - Grounded| AnswerGrader{"Answer Grader\n(Addresses question?)"}
        AnswerGrader -->|No - Irrelevant| RewriteQuery
        AnswerGrader -->|Yes - Complete| OutputAnswer([Final Answer + Citations])
        FastGen --> OutputAnswer
    end

    classDef database fill:#2980b9,stroke:#fff,stroke-width:1px,color:#fff;
    classDef process fill:#2ecc71,stroke:#fff,stroke-width:1px,color:#fff;
    classDef agent fill:#e74c3c,stroke:#fff,stroke-width:1px,color:#fff;
    classDef router fill:#f1c40f,stroke:#fff,stroke-width:1px,color:#000;
    
    class StoreFAISS database;
    class Router,GradeDocs,HallucinationGrader,AnswerGrader router;
    class DecGraph,GenerateNode,DDGWeb agent;
    class Chunking,MultiRep,Analyzer process;
```

---

## 🔄 Application Request Flow

The diagram below maps the runtime lifecycle of a user query through the system's routing decision trees and Self-RAG loops:

```mermaid
sequenceDiagram
    autonumber
    actor User as Client / CLI User
    participant Main as main.py (Runtime)
    participant Router as SemanticRouter
    participant Analyzer as QueryAnalyzer
    participant Graph as AgenticSelfRAG (LangGraph)
    participant Web as DuckDuckGo Web Search
    participant LLM as OpenAI Chat API

    User->>Main: Ask question ("Compare A and B")
    Main->>Router: Route classification
    Router-->>Main: Returns Route ("standard" / "decomposition" / "simple")

    alt Route == "simple" (Fast Path)
        Main->>LLM: Run fast_rag_chain (Skip heavy retrieval)
        LLM-->>Main: Returns direct conversational answer
        Main-->>User: Output direct answer (no source citations)
    else Route == "decomposition" / "standard" (Heavy Path)
        Main->>Analyzer: Parse metadata filters & core search term
        Analyzer-->>Main: SearchQuery object (e.g. file_type='pdf')
        
        Main->>Graph: Invoke LangGraph Agentic workflow
        Graph->>Graph: Retrieve local documents (FAISS + BM25)
        Graph->>Graph: Grade retrieved documents for relevance
        
        alt All Documents Irrelevant (CRAG Trigger)
            Graph->>Web: Query DuckDuckGo web search
            Web-->>Graph: Return search snippets
        end
        
        Graph->>LLM: Generate candidate answer from context
        LLM-->>Graph: Candidate answer text
        
        loop Double-Guardrail Verification (Max 2 retries)
            Graph->>LLM: Hallucination Grader: Is answer grounded in context?
            LLM-->>Graph: Verdict ("grounded" / "hallucination")
            
            alt Hallucination Detected
                Graph->>Graph: Rewrite search query & retrieve again
            else Answer is Grounded
                Graph->>LLM: Answer Grader: Does answer address the user query?
                LLM-->>Graph: Verdict ("yes" / "no")
                alt Answer does NOT address question
                    Graph->>Graph: Rewrite search query & retrieve again (or Web search fallback)
                end
            end
        end
        
        Graph-->>Main: Return final validated state
        Main-->>User: Display final answer + Document Citations
    end
```

---

## 🛠️ Tech Stack

* **Package Manager**: [uv](https://github.com/astral-sh/uv) (Rust-powered, ultra-fast Python environment sync)
* **Orchestration**: [LangChain](https://github.com/langchain-ai/langchain) & [LangGraph](https://github.com/langchain-ai/langgraph)
* **LLM Engine**: OpenAI API (`gpt-4o-mini` & `text-embedding-3-small`)
* **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss) (Lightweight, local, network-vulnerability-free vector library)
* **Sparse Index**: [Rank-BM25](https://github.com/dorianbrown/rank_bm25)
* **Re-ranker Model**: [Flashrank](https://github.com/prithivida/flashrank) (Quantized cross-encoder model running locally via ONNX)
* **Document Parsing**: `pypdf`, `docx2txt`, `beautifulsoup4`, `tiktoken`

---

## 📂 Project Layout

```plaintext
rag_project/
├── .env.example            # Reference configurations (Template)
├── .gitignore              # Git ignore rules for virtual environments, .env and db
├── pyproject.toml          # Modern PEP 621 project configuration managed by uv
├── requirements.txt        # Exported dependency lockfile
│
├── ingest.py               # Scraping, semantic parsing, Multi-Rep + RAPTOR building
├── query_processor.py      # Dynamic semantic embedding router & Pydantic analyzer
├── decomposition_graph.py  # LangGraph multi-hop sequential decomposition agent
├── agentic_graph.py        # LangGraph CRAG + Self-RAG Double-Guardrail agent
├── multi_rep_utils.py      # Multi-Representation Indexing utilities
├── main.py                 # CLI interface, pipeline runner, and conversational loop
└── playground.py           # Local helper script for testing embeddings & similarities
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

# Reranker selection: 'flashrank' (local ONNX Cross-Encoder) or 'cohere' (Cloud Rerank API)
RERANKER_PROVIDER=flashrank
```

### Usage

1. **Populate Documents**: Place your PDFs, Word documents (`.docx`), CSVs, or text files into the `./documents` folder.
2. **Ingest Data**: Execute the parser and chunking pipeline to build the database:
   ```bash
   # Build index using Multi-Rep indexing
   uv run ingest.py

   # Build index using Multi-Rep + RAPTOR hierarchical tree summaries
   uv run ingest.py --raptor
   ```
3. **Launch the Chat CLI**: Run the interactive conversational loop:
   ```bash
   uv run main.py
   ```

---

## 🔒 Security & Hardening

* **Local Sandbox Boundary**: The SQLite FAISS database, semantic indices, and re-ranking tasks are kept local. Document content is only sent to OpenAI for LLM inference (not for embeddings or database hosting).
* **Vulnerable Dependency Avoidance**: The project dependencies contain no external pickling/caching libraries (`diskcache`) or multi-modal faithfulness evaluation modules (`ragas`), completely eliminating vulnerabilities such as CVE-2025-69872 and CVE-2025-45691.
* **Secrets Management**: Built-in rules in `.gitignore` ensure your `.env` configuration file and local `faiss_db/` folder are never committed to version control.

---

## 📈 Performance & Scalability

* **Parallel Execution**: Retrieval queries for RAG-fusion, Multi-Query, and step-back abstractions are resolved concurrently in a `ThreadPoolExecutor` to optimize API latency.
* **In-Memory Semantic Routing**: Simple queries bypass the heavy RAG pipelines entirely, resolving in <15ms without LLM latency or token overhead.
* **Quantized Reranking**: By default, `Flashrank` utilizes local quantized ONNX weights, allowing Cross-Encoder scoring to run inside CPU constraints with negligible RAM overhead.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
