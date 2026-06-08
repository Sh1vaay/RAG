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
from query_processor import RoutingRetriever

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

    # 4. Setup Local Cross-Encoder Reranker (Flashrank)
    print("Initializing local Cross-Encoder reranking (Flashrank)...")
    try:
        # Setup Flashrank compressor to re-rank the hybrid candidates, outputting only the top 3
        compressor = FlashrankRerank(top_n=3)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=ensemble_retriever
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize Flashrank reranker: {e}", file=sys.stderr)
        return

    # 5. Setup LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 6. Setup Routing Retriever (Dynamic Translation Router)
    print("Initializing Dynamic Query Translation Router...")
    try:
        routing_retriever = RoutingRetriever(
            base_retriever=compression_retriever,
            llm=llm
        )
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
                # Invoke conversational multi-query RAG chain
                response = rag_chain.invoke({
                    "input": query, 
                    "chat_history": chat_history
                })
                
                answer = response["answer"]
                print(f"\nAnswer:\n{answer}")
                
                # Extract and format sources (post-reranked and de-duplicated)
                context_docs = response.get("context", [])
                if context_docs:
                    print("\n📚 Top Sources (De-duplicated & Re-ranked):")
                    for idx, doc in enumerate(context_docs, 1):
                        source_url = doc.metadata.get("source", "Unknown Source")
                        title = doc.metadata.get("title", "Document")
                        snippet = doc.page_content[:150].strip().replace('\n', ' ')
                        print(f"  [{idx}] {title} - {source_url}")
                        print(f"      Snippet preview: \"{snippet}...\"")
                
                # Append both question and answer to stateful history
                chat_history.append(HumanMessage(content=query))
                chat_history.append(AIMessage(content=answer))
                
            except Exception as e:
                print(f"\n[ERROR] An error occurred while invoking the model: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred in loop: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
