import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from query_processor import RoutingRetriever, QueryAnalyzer, SearchQuery
from multi_rep_utils import restore_original_content

def post_filter_documents(docs: list, query: SearchQuery) -> list:
    """Bulletproof post-filtering layer to ensure all retrieved documents strictly match constraints."""
    filtered_docs = []
    for doc in docs:
        # Check file_type
        if query.file_type and doc.metadata.get("file_type") != query.file_type:
            continue
        # Check year
        if query.publish_year and doc.metadata.get("year") != query.publish_year:
            continue
        # Check page number
        if query.page_number and doc.metadata.get("page") != query.page_number:
            continue
        # Check data_source
        if query.data_source and doc.metadata.get("data_source") != query.data_source:
            continue
        filtered_docs.append(doc)
    return filtered_docs

# Load environment variables
load_dotenv()

def main():
    # 1. Validation Checks
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("[ERROR] OPENAI_API_KEY is not set or is still the default placeholder value in .env.", file=sys.stderr)
        print("Please configure your OpenAI API key in the .env file before running main.py.", file=sys.stderr)
        return

    # Check if vectorstore exists
    db_path = "./chroma_db"
    if not os.path.exists(db_path) or not os.listdir(db_path):
        print(f"[ERROR] Chroma database directory '{db_path}' is empty or does not exist.", file=sys.stderr)
        print("Please run data ingestion first by running: `uv run ingest.py`", file=sys.stderr)
        return

    # 2. Load the existing Chroma database from disk
    print("Loading database...")
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
    except Exception as e:
        print(f"[ERROR] Failed to load vector database: {e}", file=sys.stderr)
        return
    
    # 3. Setup Keyword (BM25) and Semantic (Chroma) Retrievers
    print("Initializing hybrid retrieval (BM25 + Chroma)...")
    try:
        # Extract all documents currently persisted inside the vector store to seed the BM25 index
        raw_db_data = vectorstore.get()
        documents = []
        if raw_db_data and "documents" in raw_db_data:
            from langchain_core.documents import Document
            for text, meta in zip(raw_db_data["documents"], raw_db_data["metadatas"]):
                documents.append(Document(page_content=text, metadata=meta))
        
        if not documents:
            raise ValueError("No documents loaded from vector store database to build BM25 search index.")

        # Initialize keyword search on exact document contents
        bm25_retriever = BM25Retriever.from_documents(documents)
        # Retrieve a broader pool of candidates (k=8) to allow the reranker to prune
        bm25_retriever.k = 8

        # Initialize semantic/vector search (k=8)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

        # Combine retrievers using Reciprocal Rank Fusion (RRF)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )
    except Exception as e:
        print(f"[ERROR] Failed to build hybrid retriever: {e}", file=sys.stderr)
        return

    # 4. Setup Reranker (Flashrank or Cohere)
    reranker_provider = os.getenv("RERANKER_PROVIDER", "flashrank").strip().lower()
    try:
        if reranker_provider == "cohere":
            cohere_api_key = os.getenv("COHERE_API_KEY")
            if not cohere_api_key or cohere_api_key == "your_cohere_api_key_here":
                print("[WARNING] COHERE_API_KEY is not set. Falling back to local Flashrank.", file=sys.stderr)
                compressor = FlashrankRerank(top_n=3)
                print("Initializing local Cross-Encoder reranking (Flashrank)...")
            else:
                from langchain_cohere import CohereRerank
                compressor = CohereRerank(top_n=3)
                print("Initializing cloud-based Cohere reranking...")
        else:
            compressor = FlashrankRerank(top_n=3)
            print("Initializing local Cross-Encoder reranking (Flashrank)...")

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=ensemble_retriever
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize reranker ({reranker_provider}): {e}", file=sys.stderr)
        return

    # 5. Setup LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    query_analyzer = QueryAnalyzer(llm)

    # 6. Setup Routing Retriever (Dynamic Translation Router)
    print("Initializing Dynamic Query Translation Router...")
    try:
        routing_method = os.getenv("ROUTING_METHOD", "semantic").strip().lower()
        if routing_method not in ("semantic", "llm"):
            routing_method = "semantic"
            
        routing_retriever = RoutingRetriever(
            base_retriever=compression_retriever,
            llm=llm,
            embeddings=embeddings,
            routing_method=routing_method
        )
        routing_retriever.initialize_router()
    except Exception as e:
        print(f"[ERROR] Failed to initialize Routing Retriever: {e}", file=sys.stderr)
        return

    # 7. Define Prompts for Conversational Flow
    
    # Prompt to reformulate follow-up questions to be standalone
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Prompt for final response generation
    qa_system_prompt = (
        "Answer the question based only on the following context:\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # 8. Build the Chains
    
    # Create retriever that takes history into account and pipes to the routed hybrid retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, routing_retriever, contextualize_q_prompt
    )

    # Create QA chain that combines retrieved documents
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create full retrieval RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Initialize in-memory session chat history
    chat_history = []

    # 9. Execution Loop
    print("\n=======================================================")
    print("🚀 Conversational Multi-Query Hybrid RAG Ready.")
    print("Type 'exit' or 'quit' to close.")
    print("=======================================================")
    
    try:
        while True:
            try:
                query = input("\nAsk a question: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting. Goodbye!")
                break
                
            if not query:
                continue
                
            if query.lower() in ('exit', 'quit'):
                print("Goodbye!")
                break
                
            print("Thinking...")
            try:
                # 1. Contextualize query if history exists
                if chat_history:
                    contextualize_chain = contextualize_q_prompt | llm
                    standalone_q = contextualize_chain.invoke({
                        "input": query,
                        "chat_history": chat_history
                    }).content.strip()
                else:
                    standalone_q = query
                
                # 2. Extract query filters/constraints
                structured_query = query_analyzer.analyze(standalone_q)
                
                # Build Chroma database filter dictionary
                chroma_filters = {}
                if structured_query.file_type:
                    chroma_filters["file_type"] = structured_query.file_type
                if structured_query.publish_year:
                    chroma_filters["year"] = structured_query.publish_year
                if structured_query.page_number:
                    chroma_filters["page"] = structured_query.page_number
                if structured_query.data_source:
                    chroma_filters["data_source"] = structured_query.data_source
                
                # Dynamically inject filter to the Chroma retriever
                vector_retriever.search_kwargs["filter"] = chroma_filters if chroma_filters else None
                if chroma_filters:
                    print(f"🎯 [Query Analyzer] Extracted constraints: {chroma_filters}")
                
                # 3. Determine Route
                route, _ = routing_retriever.determine_route(structured_query.content_search)
                
                # 4. Execute Route
                if route == "decomposition":
                    from decomposition_graph import create_decomposition_graph
                    # Build and execute the cyclic LangGraph decomposition agent
                    graph = create_decomposition_graph(compression_retriever, llm)

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

                elif route in ("standard",) and (
                    any(kw in structured_query.content_search.lower() for kw in
                        ("compare", "versus", "difference", "evaluate", "analyse", "analyze",
                         "pros and cons", "tradeoff", "contrast", "critique", "assessment"))
                ):
                    # ── Agentic CRAG + Self-RAG path (complex analytical queries) ──
                    print("🤖 [Agentic RAG] Complex query detected. Activating CRAG + Self-RAG loop...")
                    from agentic_graph import create_agentic_graph
                    agentic = create_agentic_graph(compression_retriever, llm)
                    agentic_state = agentic.invoke({
                        "question": structured_query.content_search,
                        "rewritten_question": "",
                        "retrieved_docs": [],
                        "relevant_docs": [],
                        "answer": "",
                        "reflection_passed": False,
                        "retry_count": 0,
                    })
                    answer = agentic_state["answer"]
                    context_docs = restore_original_content(
                        agentic_state["relevant_docs"] or agentic_state["retrieved_docs"]
                    )

                else:
                    # Retrieve documents using chosen translation strategy
                    context_docs = routing_retriever.retrieve_for_route(structured_query.content_search, route)

                    # Apply final post-filtering layer to ensure perfect matches (especially for BM25)
                    context_docs = post_filter_documents(context_docs, structured_query)

                    # Restore original content (swapped out during Multi-Rep indexing)
                    context_docs = restore_original_content(context_docs)

                    # Generate final answer using context
                    answer = question_answer_chain.invoke({
                        "context": context_docs,
                        "input": structured_query.content_search,
                        "chat_history": chat_history
                    })
                
                # 4. Display Output
                print(f"\nAnswer:\n{answer}")
                
                if context_docs:
                    print("\n📚 Top Sources (De-duplicated & Re-ranked):")
                    # De-duplicate context docs for print preview
                    seen_sources = set()
                    unique_sources = []
                    for doc in context_docs:
                        key = (doc.page_content[:100], doc.metadata.get("source", ""))
                        if key not in seen_sources:
                            seen_sources.add(key)
                            unique_sources.append(doc)
                            
                    for idx, doc in enumerate(unique_sources[:5], 1):
                        source_url = doc.metadata.get("source", "Unknown Source")
                        title = doc.metadata.get("title", "Document")
                        snippet = doc.page_content[:150].strip().replace('\n', ' ')
                        print(f"  [{idx}] {title} - {source_url}")
                        print(f"      Snippet preview: \"{snippet}...\"")
                
                # 5. Update Conversation History
                chat_history.append(HumanMessage(content=query))
                chat_history.append(AIMessage(content=answer))
                
            except Exception as e:
                print(f"\n[ERROR] An error occurred while invoking the model: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred in loop: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
