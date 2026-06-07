import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

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
    
    # 3. Setup retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. Setup LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 5. Define Prompts for Conversational Flow
    
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

    # 6. Build the Chains
    
    # Create retriever that takes history into account
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Create QA chain that combines retrieved documents
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create full retrieval RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Initialize in-memory session chat history
    chat_history = []

    # 7. Execution Loop
    print("\n=======================================================")
    print("🚀 Conversational RAG System Ready. Type 'exit' or 'quit' to close.")
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
                # Invoke conversational RAG chain
                response = rag_chain.invoke({
                    "input": query, 
                    "chat_history": chat_history
                })
                
                answer = response["answer"]
                print(f"\nAnswer:\n{answer}")
                
                # Append both question and answer to stateful history
                chat_history.append(HumanMessage(content=query))
                chat_history.append(AIMessage(content=answer))
                
            except Exception as e:
                print(f"\n[ERROR] An error occurred while invoking the model: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred in loop: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
