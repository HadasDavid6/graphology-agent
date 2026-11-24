"""RAG layer — local lexical retrieval over a corpus of documents.

Minimal, offline, no dependencies. Swap LexicalIndex for a vector index later without
changing the agent. See store.py.
"""

from core.tools.rag.store import Chunk, EmbeddingIndex, chunk_text, load_corpus, tokenize

LexicalIndex = EmbeddingIndex

__all__ = ["Chunk", "EmbeddingIndex", "LexicalIndex", "chunk_text", "load_corpus", "tokenize"]
