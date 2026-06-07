import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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
        # Use the same embedding model defined in ingest.py to ensure vector space matching
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

    # 4. Define Prompt
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 5. Setup LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 6. Build the Chain (LCEL)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 7. Execution Loop
    print("\n=======================================================")
    print("🚀 RAG System Ready. Type 'exit' or 'quit' to close.")
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
                response = rag_chain.invoke(query)
                print(f"\nAnswer:\n{response}")
            except Exception as e:
                print(f"\n[ERROR] An error occurred while invoking the model: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred in loop: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
