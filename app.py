import os
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

import transcript
from rag import FAISS, ChatGroq, build_vector_store, get_embeddings, query_groq, search_context
from transcript import YouTubeTranscriptApi, chunk_transcript, extract_video_id

load_dotenv()

app = Flask(__name__)

# ponytail: single in-memory vector store; use session-id or external store if multi-tenant needed
VECTOR_STORE = None


def fetch_transcript(video_id: str, api_cls=None):
    """Fetch transcript delegating to transcript module with app's YouTubeTranscriptApi by default."""
    return transcript.fetch_transcript(video_id, api_cls=api_cls or YouTubeTranscriptApi)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/load", methods=["POST"])
def load_video():
    global VECTOR_STORE
    data = request.get_json(silent=True)
    url = data.get("url") if isinstance(data, dict) else None
    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({"error": "Invalid or missing YouTube URL"}), 400

    try:
        text = fetch_transcript(video_id, api_cls=YouTubeTranscriptApi)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch transcript: {str(e)}"}), 400

    chunks = chunk_transcript(text)
    if not chunks:
        return jsonify({"error": "No transcript content found"}), 400

    try:
        VECTOR_STORE = build_vector_store(chunks)
    except Exception as e:
        return jsonify({"error": f"Failed to index transcript: {str(e)}"}), 500

    return jsonify({"message": "Transcript indexed successfully", "chunks": len(chunks), "video_id": video_id})


@app.route("/chat", methods=["POST"])
def chat():
    global VECTOR_STORE
    if VECTOR_STORE is None:
        return jsonify({"error": "No video transcript loaded yet"}), 400

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        data = {}

    query = data.get("query")
    query = query.strip() if isinstance(query, str) else ""
    if not query:
        return jsonify({"error": "Query is required"}), 400

    api_key = data.get("api_key")
    api_key = api_key.strip() if isinstance(api_key, str) else ""
    api_key = api_key or (os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key:
        return jsonify({"error": "Groq API key is required"}), 400

    try:
        context, count = search_context(VECTOR_STORE, query, k=4)
        answer = query_groq(context, query, api_key, llm_cls=ChatGroq)
        return jsonify({"answer": answer, "context_chunks": count})
    except Exception as e:
        return jsonify({"error": f"LLM error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
