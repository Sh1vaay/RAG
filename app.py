import os
import sys
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from langchain_core.messages import HumanMessage, AIMessage

# Import pipeline components from main.py
from main import setup_pipeline, post_filter_documents
from multi_rep_utils import restore_original_content

# Initialize FastAPI application
app = FastAPI(
    title="Conversational RAG API",
    description="Backend API serving the Advanced Local Conversational RAG pipeline",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline state container
pipeline = None

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

@app.on_event("startup")
def startup_event():
    global pipeline
    try:
        pipeline = setup_pipeline()
        print("💡 [API] RAG Pipeline loaded successfully.")
    except Exception as e:
        print(f"❌ [API] Failed to initialize RAG Pipeline: {e}", file=sys.stderr)
        # We don't raise here to allow server startup and let /status show the error

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

    return {
        "status": "ready" if pipeline else "error",
        "database_loaded": db_exists,
        "document_chunks": doc_count,
        "routing_method": os.getenv("ROUTING_METHOD", "semantic"),
        "reranker_provider": os.getenv("RERANKER_PROVIDER", "flashrank"),
        "openai_api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        "langsmith_tracing": os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    }

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
            snippet = doc.page_content[:200].strip()
            
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload new files into the local documents directory."""
    os.makedirs("./documents", exist_ok=True)
    saved_files = []
    for file in files:
        file_path = os.path.join("./documents", file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        saved_files.append(file.filename)
        
    return {"status": "success", "uploaded_files": saved_files}

@app.post("/api/ingest")
async def trigger_ingestion(raptor: bool = False):
    """Triggers the document ingestion and indexing pipeline."""
    try:
        import subprocess
        cmd = [sys.executable, "ingest.py"]
        if raptor:
            cmd.append("--raptor")
            
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Reload pipeline after successful ingestion
        global pipeline
        pipeline = setup_pipeline()
        
        return {
            "status": "success", 
            "message": "Ingestion completed and DB index reloaded.",
            "output": result.stdout[:500]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
