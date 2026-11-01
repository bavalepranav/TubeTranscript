import os
from langchain_community.vectorstores.faiss import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDINGS = None


def get_embeddings():
    """Retrieve or initialize the HuggingFace embeddings model."""
    global EMBEDDINGS
    if EMBEDDINGS is None:
        try:
            EMBEDDINGS = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"local_files_only": True},
            )
        except Exception:
            EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return EMBEDDINGS


def build_vector_store(chunks: list[str]) -> FAISS:
    """Index text chunks into an in-memory FAISS vector store."""
    if not chunks:
        raise ValueError("Cannot build vector store from empty chunks")
    return FAISS.from_texts(chunks, get_embeddings())


def search_context(vector_store: FAISS, query: str, k: int = 4) -> tuple[str, int]:
    """Retrieve top-k relevant chunks from FAISS and combine them into a context string."""
    docs = vector_store.similarity_search(query, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)
    return context, len(docs)


def query_groq(context: str, query: str, api_key: str, model_name: str = None, llm_cls=None) -> str:
    """Send RAG prompt to Groq LLM and return clean string response."""
    if llm_cls is None:
        llm_cls = ChatGroq

    if not model_name:
        model_name = (os.getenv("GROQ_MODEL") or "").strip()
    if not model_name or model_name == "llama-3.3-70b-versatile":
        model_name = "openai/gpt-oss-120b"

    prompt = (
        f"You are a helpful assistant answering questions about a YouTube video transcript.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    llm = llm_cls(groq_api_key=api_key, model_name=model_name)
    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        return "".join(p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in content)
    return str(content)

