from contextlib import asynccontextmanager
import os
import sys
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel


# Import pipeline components from main.py
from main import setup_pipeline, post_filter_documents
from multi_rep_utils import restore_original_content

# Global pipeline state container
pipeline = None

# Modern Lifespan Manager replacing deprecated startup event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    try:
        pipeline = setup_pipeline()
        print("💡 [API] RAG Pipeline loaded successfully.")
    except Exception as e:
        print(f"❌ [API] Failed to initialize RAG Pipeline: {e}", file=sys.stderr)
    yield

# Initialize FastAPI application
app = FastAPI(
    title="Conversational RAG API",
    description="Backend API serving the Advanced Local Conversational RAG pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration (CORS origins configurable via env)
allowed_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

class SourceDocument(BaseModel):
    title: str
    source: str
    page: Optional[int] = None
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    route: str
    sources: List[SourceDocument]

class ConfigUpdateRequest(BaseModel):
    routing_method: str
    reranker_provider: str
    openai_key: Optional[str] = None
    cohere_key: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serves the main single-page dashboard HTML application."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return (
            "<html>"
            "<head><title>Aether AI - Loading</title></head>"
            "<body style=\"background:#0f172a;color:#fff;font-family:sans-serif;"
            "display:flex;align-items:center;justify-content:center;height:100vh;\">"
            "<div>"
            "<h2>Aether AI Dashboard</h2>"
            "<p>Frontend file <code>index.html</code> not found. "
            "Please create it in the workspace root.</p>"
            "</div>"
            "</body>"
            "</html>"
        )

@app.get("/api/status")
def get_status():
    """Returns database status, active configuration, and loader metrics."""
    db_path = "./faiss_db"
    db_exists = os.path.exists(db_path) and len(os.listdir(db_path)) > 0
    doc_count = 0
    if db_exists and pipeline:
        try:
            doc_count = len(pipeline["vector_retriever"].vectorstore.docstore._dict)
        except Exception:
            pass

    # Scan documents folder for staged files list
    staged_files = []
    if os.path.exists("./documents"):
        for f in os.listdir("./documents"):
            if os.path.isfile(os.path.join("./documents", f)):
                size = os.path.getsize(os.path.join("./documents", f))
                mbytes = size / (1024 * 1024)
                kbytes = size / 1024
                size_str = f"{mbytes:.2f} MB" if size > 1024 * 1024 else f"{kbytes:.1f} KB"
                staged_files.append({
                    "name": f,
                    "size": size_str,
                    "status": "ready"
                })

    return {
        "status": "ready" if pipeline else "error",
        "database_loaded": db_exists,
        "document_chunks": doc_count,
        "routing_method": os.getenv("ROUTING_METHOD", "semantic"),
        "reranker_provider": os.getenv("RERANKER_PROVIDER", "flashrank"),
        "openai_api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        "langsmith_tracing": os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
        "staged_files": staged_files
    }

@app.post("/api/config")
def update_config(config: ConfigUpdateRequest):
    """Dynamically updates environment configurations and reloads the pipeline if necessary."""
    global pipeline
    # Validate input settings to prevent dynamic configuration pollution
    if config.routing_method not in ("semantic", "llm"):
        raise HTTPException(
            status_code=400,
            detail="Invalid routing_method. Must be 'semantic' or 'llm'."
        )
    if config.reranker_provider not in ("flashrank", "cohere"):
        raise HTTPException(
            status_code=400,
            detail="Invalid reranker_provider. Must be 'flashrank' or 'cohere'."
        )

    os.environ["ROUTING_METHOD"] = config.routing_method
    os.environ["RERANKER_PROVIDER"] = config.reranker_provider
    if config.openai_key and config.openai_key.strip():
        os.environ["OPENAI_API_KEY"] = config.openai_key
    if config.cohere_key and config.cohere_key.strip():
        os.environ["COHERE_API_KEY"] = config.cohere_key

    # Reload components to pick up new routing/reranking parameters
    try:
        pipeline = setup_pipeline()
        return {
            "status": "success",
            "message": "Configuration updated and pipeline re-initialized."
        }
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Pipeline reload failed. Check server configuration logs."
        )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline is not initialized.")

    try:
        query = request.message.strip()

        # 1. Convert JSON chat history to LangChain messages
        chat_history = []
        for msg in request.history:
            if msg.role == "user":
                chat_history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                chat_history.append(AIMessage(content=msg.content))

        # 2. Contextualize query if history exists
        llm = pipeline["llm"]
        if chat_history:
            contextualize_chain = pipeline["contextualize_q_prompt"] | llm
            standalone_q = contextualize_chain.invoke({
                "input": query,
                "chat_history": chat_history
            }).content.strip()
        else:
            standalone_q = query
            
        # 3. Check Semantic Router
        routing_retriever = pipeline["routing_retriever"]
        route, _ = routing_retriever.determine_route(standalone_q)
        
        sources_list = []
        
        if route == "simple":
            # Fast Path Bypass
            fast_result = pipeline["fast_rag_chain"].invoke({
                "input": standalone_q,
                "chat_history": chat_history
            })
            answer = fast_result["answer"]
            context_docs = fast_result.get("context", [])
        else:
            # Heavy pipeline
            query_analyzer = pipeline["query_analyzer"]
            structured_query = query_analyzer.analyze(standalone_q)
            
            # Build DB filters
            db_filters = {}
            if structured_query.file_type:
                db_filters["file_type"] = structured_query.file_type
            if structured_query.publish_year:
                db_filters["year"] = structured_query.publish_year
            if structured_query.page_number:
                db_filters["page"] = structured_query.page_number
            if structured_query.data_source:
                db_filters["data_source"] = structured_query.data_source
                
            pipeline["vector_retriever"].search_kwargs["filter"] = db_filters if db_filters else None
            
            if route == "decomposition":
                from decomposition_graph import create_decomposition_graph
                graph = create_decomposition_graph(pipeline["compression_retriever"], llm)
                state = graph.invoke({
                    "main_question": structured_query.content_search,
                    "sub_questions": [],
                    "current_index": 0,
                    "sub_answers": [],
                    "retrieved_docs": [],
                    "final_answer": ""
                })
                answer = state["final_answer"]
                context_docs = restore_original_content(state["retrieved_docs"])
                
            elif route == "standard" and any(
                kw in structured_query.content_search.lower() for kw in
                ("compare", "versus", "difference", "evaluate", "analyse", "analyze", "pros and cons", "tradeoff", "contrast")
            ):
                from agentic_graph import create_agentic_graph
                agentic = create_agentic_graph(pipeline["compression_retriever"], llm)
                agentic_state = agentic.invoke({
                    "question": structured_query.content_search,
                    "rewritten_question": "",
                    "retrieved_docs": [],
                    "relevant_docs": [],
                    "answer": "",
                    "reflection_passed": False,
                    "answer_relevant": False,
                    "retry_count": 0,
                })
                answer = agentic_state["answer"]
                context_docs = restore_original_content(
                    agentic_state["relevant_docs"] or agentic_state["retrieved_docs"]
                )
            else:
                # Retrieve documents
                context_docs = routing_retriever.retrieve_for_route(structured_query.content_search, route)
                context_docs = post_filter_documents(context_docs, structured_query)
                context_docs = restore_original_content(context_docs)
                
                answer = pipeline["question_answer_chain"].invoke({
                    "context": context_docs,
                    "input": structured_query.content_search,
                    "chat_history": chat_history
                })

        # Process and de-duplicate citations
        seen_keys = set()
        for doc in context_docs:
            source_url = doc.metadata.get("source", "Unknown Source")
            title = doc.metadata.get("title", os.path.basename(source_url))
            page = doc.metadata.get("page")
            snippet = doc.page_content[:250].strip()
            
            key = (title, page, snippet[:50])
            if key not in seen_keys:
                seen_keys.add(key)
                sources_list.append(SourceDocument(
                    title=title,
                    source=source_url,
                    page=page,
                    snippet=snippet
                ))

        return ChatResponse(
            answer=answer,
            route=route,
            sources=sources_list[:5]
        )

    except Exception as e:
        print(f"[API ERROR] Chat execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred while processing your chat request. Please check server logs."
        )

@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload new files into the local documents directory with path traversal sanitization."""
    os.makedirs("./documents", exist_ok=True)
    saved_files = []
    for file in files:
        # Sanitise file name to prevent path traversal (CWE-22 / CWE-23)
        safe_filename = os.path.basename(file.filename)
        safe_filename = safe_filename.replace("\0", "").replace("/", "").replace("\\", "")
        
        # Ensure name is not empty after sanitization
        if not safe_filename:
            continue
            
        file_path = os.path.join("./documents", safe_filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        saved_files.append(safe_filename)
        
    return {"status": "success", "uploaded_files": saved_files}

@app.post("/api/ingest")
async def trigger_ingestion(raptor: bool = False):
    """Triggers the document ingestion and indexing pipeline asynchronously."""
    try:
        import asyncio
        cmd = [sys.executable, "ingest.py"]
        if raptor:
            cmd.append("--raptor")
            
        # Run subprocess asynchronously
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            err_msg = stderr.decode().strip()
            print(f"[API ERROR] Ingestion subprocess failed: {err_msg}", file=sys.stderr)
            raise RuntimeError(f"Ingestion process exited with code {proc.returncode}")
            
        # Reload pipeline after successful ingestion
        global pipeline
        pipeline = setup_pipeline()
        
        return {
            "status": "success", 
            "message": "Ingestion completed and DB index reloaded.",
            "output": stdout.decode()[:1000]
        }
    except Exception as e:
        print(f"[API ERROR] Ingestion failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred during document ingestion. Please check server logs."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
