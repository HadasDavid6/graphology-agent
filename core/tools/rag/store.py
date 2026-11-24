"""Local RAG store — semantic embeddings retrieval, pure Python, no dependencies.

This module loads md/txt/pdf files, chunks them, and uses OpenRouter embeddings to
build a semantic vector index. The index is cached locally so the corpus only needs
one-time embedding setup.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from core.providers.openrouter import OpenRouterProvider

# matches Latin, Hebrew, and digit runs as tokens
_TOKEN = re.compile(r"[A-Za-z\u0590-\u05FF0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def chunk_text(text: str, max_chars: int = 700) -> list[str]:
    """Split on blank lines into paragraphs, then merge paragraphs up to max_chars.
    Keeps manual sections intact rather than cutting mid-sentence."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur:
        chunks.append(cur)
    return chunks


@dataclass
class Chunk:
    doc: str
    text: str
    tokens: list[str] = field(default_factory=list)


def extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF file using pypdf."""
    reader = PdfReader(path)
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def load_corpus(corpus_dir: str | Path) -> list[Chunk]:
    """Load .md/.txt/.pdf files from a directory into tokenized chunks."""
    chunks: list[Chunk] = []
    for path in sorted(Path(corpus_dir).glob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
        elif path.suffix.lower() == ".pdf":
            text = extract_pdf_text(path)
        else:
            continue
        if not text.strip():
            continue
        for c in chunk_text(text):
            chunks.append(Chunk(doc=path.stem, text=c, tokens=tokenize(c)))
    return chunks


def _corpus_hash(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.doc.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / ((norm_a**0.5) * (norm_b**0.5))


class EmbeddingIndex:
    """Semantic vector index over chunks."""

    def __init__(self, chunks: list[Chunk], embeddings: list[list[float]], embed_model: str):
        self.chunks = chunks
        self.embeddings = embeddings
        self.embed_model = embed_model
        self.dim = len(embeddings[0]) if embeddings else 0

    @classmethod
    def from_corpus(cls, corpus_dir: str | Path,
                    embed_model: str = "text-embedding-3-large",
                    cache: bool = True) -> "EmbeddingIndex":
        chunks = load_corpus(corpus_dir)
        if not chunks:
            return cls([], [], embed_model)

        cache_path = Path(corpus_dir) / ".rag_embeddings.json"
        corpus_hash = _corpus_hash(chunks)

        if cache and cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                if (data.get("hash") == corpus_hash and
                        data.get("model") == embed_model and
                        isinstance(data.get("embeddings"), list) and
                        len(data["embeddings"]) == len(chunks)):
                    return cls(chunks, data["embeddings"], embed_model)
            except (json.JSONDecodeError, OSError):
                pass

        provider = OpenRouterProvider(model=embed_model)
        inputs = [chunk.text for chunk in chunks]
        embeddings = provider.embed(inputs, model=embed_model)

        if cache:
            try:
                cache_path.write_text(
                    json.dumps({
                        "hash": corpus_hash,
                        "model": embed_model,
                        "embeddings": embeddings,
                    }, ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError:
                pass

        return cls(chunks, embeddings, embed_model)

    def search(self, query: str, provider: OpenRouterProvider, k: int = 4) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []

        query_embed = provider.embed([query], model=self.embed_model)[0]
        scored: list[tuple[float, int]] = []
        for i, embedding in enumerate(self.embeddings):
            score = _cosine_similarity(query_embed, embedding)
            if score > 0.0:
                scored.append((score, i))

        scored.sort(reverse=True)
        return [(self.chunks[i], score) for score, i in scored[:k]]
