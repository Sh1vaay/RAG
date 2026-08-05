"""
Internal assistant interface. Runs as a Streamlit app if streamlit is
installed, otherwise falls back to a simple REPL - both call the exact
same retriever + guardrail logic underneath.

Run: streamlit run src/app.py
Or:  python -m src.app   (CLI fallback)
"""
from .retriever import Retriever, ask


def run_cli():
    print("Internal Support Assistant (CLI mode) - type 'quit' to exit\n")
    retriever = Retriever()
    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue
        answer = ask(query, retriever=retriever)
        confidence_note = "" if answer.confident else "  [LOW CONFIDENCE]"
        print(f"Assistant: {answer.text}{confidence_note}")
        if answer.sources:
            print(f"  Sources: {', '.join(answer.sources)}")
        print()


def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="Internal Support Assistant", page_icon="🤖")
    st.title("🤖 Internal Support Assistant")
    st.caption("Internal tool - answers are grounded only in company documents. Not for client use.")

    if "retriever" not in st.session_state:
        st.session_state.retriever = Retriever()

    if "history" not in st.session_state:
        st.session_state.history = []

    query = st.chat_input("Ask a question about our services...")

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn["sources"]:
                st.caption(f"Sources: {', '.join(turn['sources'])}")
            if not turn["confident"]:
                st.warning("Low confidence - please verify manually.")

    if query:
        answer = ask(query, retriever=st.session_state.retriever)
        st.session_state.history.append({
            "query": query,
            "answer": answer.text,
            "sources": answer.sources,
            "confident": answer.confident,
        })
        st.rerun()


if __name__ == "__main__":
    try:
        import streamlit  # noqa: F401
        run_streamlit()
    except ImportError:
        run_cli()
