# 🎥 YouTube Video Q&A Chatbot

Ask questions about any YouTube video and get answers based on its transcript, using RAG (Retrieval-Augmented Generation).

## How It Works

1. Paste a YouTube link and the app fetches the video's transcript.
2. The transcript is split into small overlapping chunks.
3. The chunks are converted to embeddings and stored in a FAISS index.
4. When you ask a question, the 4 most relevant chunks are retrieved.
5. Those chunks and your question are sent to a Groq LLM, which writes the answer.

## Tech Stack

| Part | Tool |
|---|---|
| Web framework | Flask |
| Transcripts | youtube-transcript-api |
| Text splitting | LangChain text splitters |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector store | FAISS |
| LLM | Groq (via langchain-groq) |

## Project Structure

```
├── app.py              # Flask routes
├── transcript.py       # Fetch and chunk transcripts
├── rag.py              # Embeddings, search, Groq query
├── templates/
│   └── index.html      # Web page
├── requirements.txt    # Dependencies
├── .env.example        # Example settings
└── README.md
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file (see `.env.example`):
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=openai/gpt-oss-120b
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open http://127.0.0.1:5000, paste a YouTube link, load it, and ask questions.

## Note

Only one video is stored at a time, and the video must have captions.
