from urllib.parse import parse_qs, unquote, urlparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url: str):
    """Extract YouTube video ID from various URL formats."""
    if not isinstance(url, str):
        return None
    url = url.strip("<>\"' \t\r\n")
    if not url:
        return None
    if not url.startswith(("http://", "https://", "//")):
        url = "https://" + url
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()

    def clean_id(raw_id: str):
        if not raw_id:
            return None
        val = unquote(raw_id).strip()
        return val if val and not any(c in val for c in " \t\r\n/?&#") else None

    if hostname in ("youtube.com", "youtube-nocookie.com") or hostname.endswith((".youtube.com", ".youtube-nocookie.com")):
        v_list = parse_qs(parsed.query).get("v")
        if v_list and clean_id(v_list[0]):
            return clean_id(v_list[0])
        parts = [p.strip() for p in parsed.path.split("/") if p.strip()]
        if len(parts) >= 2 and parts[0] in ("embed", "shorts", "live", "v", "watch"):
            return clean_id(parts[1])
    elif hostname == "youtu.be" or hostname.endswith(".youtu.be"):
        parts = [p.strip() for p in parsed.path.split("/") if p.strip()]
        if parts:
            return clean_id(parts[0])
    return None


def fetch_transcript(video_id: str, api_cls=None) -> str:
    """Fetch transcript text for a YouTube video with language fallback."""
    if api_cls is None:
        api_cls = YouTubeTranscriptApi

    if hasattr(api_cls, "get_transcript"):
        raw = api_cls.get_transcript(video_id)
    else:
        api = api_cls()
        try:
            raw = api.fetch(video_id)
        except Exception:
            if hasattr(api, "list"):
                transcripts = api.list(video_id)
                first = next(iter(transcripts), None)
                if not first:
                    raise RuntimeError(f"No transcripts found for video {video_id}")
                raw = first.fetch()
            else:
                raise
    parts = []
    for item in raw:
        if hasattr(item, "text"):
            parts.append(item.text)
        elif isinstance(item, dict) and "text" in item:
            parts.append(item["text"])
        else:
            parts.append(str(item))
    return " ".join(parts)


def chunk_transcript(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list[str]:
    """Split transcript text into overlapping chunks for vector embedding."""
    if not text or not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

