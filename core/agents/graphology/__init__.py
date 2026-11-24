from __future__ import annotations

from pathlib import Path
import yaml

from core.tools.rag import EmbeddingIndex

_DIR = Path(__file__).parent
_CFG = yaml.safe_load((_DIR / "agent.yaml").read_text(encoding="utf-8"))
_RAG = _CFG.get("rag", {}) or {}
_raw_corpus_dir = _RAG.get("corpus_dir", "corpus")
_CORPUS_DIR = Path(_raw_corpus_dir) if Path(_raw_corpus_dir).is_absolute() else _DIR / _raw_corpus_dir
_EMBED_MODEL = _RAG.get("embed_model", "text-embedding-3-large")

# Load the corpus once when the package imports.
GRAPH_INDEX = EmbeddingIndex.from_corpus(_CORPUS_DIR, embed_model=_EMBED_MODEL)

__all__ = ["GRAPH_INDEX", "_CORPUS_DIR", "_EMBED_MODEL"]
