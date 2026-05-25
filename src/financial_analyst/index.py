"""Vector index construction, persistence, and reload."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

from .config import EMBED_MODEL_NAME, EMBED_MODEL_REVISION, configure_settings
from .ingest import ingest_all
from .manifest import manifest_hash
from .paths import STORAGE_DIR

logger = logging.getLogger(__name__)


# Defaults — used by Lab 7.1 baseline. Tuned by Lab 7.3.
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50


@dataclass
class IndexConfig:
    """All knobs that affect what gets embedded."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    embed_model: str = EMBED_MODEL_NAME
    embed_revision: str = EMBED_MODEL_REVISION

    def cache_key(self) -> str:
        """Short stable id used to namespace ``storage/`` directories."""
        return (
            f"c{self.chunk_size}_o{self.chunk_overlap}"
            f"_{self.embed_model.split('/')[-1]}"
            f"_{self.embed_revision[:8]}"
            f"_{manifest_hash()}"
        )

    def storage_dir(self) -> Path:
        return STORAGE_DIR / self.cache_key()


def build_index(
    cfg: IndexConfig | None = None,
    force: bool = False,
) -> VectorStoreIndex:
    """Ingest the manifest, chunk, embed, and persist a vector index.

    Idempotent: if a persisted index for this config already exists,
    it is loaded rather than rebuilt — unless ``force=True``.
    """
    cfg = cfg or IndexConfig()
    configure_settings(llm=None)  # embedding only — no LLM call during build
    target = cfg.storage_dir()

    if target.exists() and not force:
        logger.info("Loading cached index from %s", target)
        return load_index(cfg)

    if target.exists() and force:
        shutil.rmtree(target)

    logger.info(
        "Building index in %s (chunk_size=%d, overlap=%d)",
        target, cfg.chunk_size, cfg.chunk_overlap,
    )
    documents, _reports = ingest_all()
    splitter = SentenceSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    logger.info(
        "Embedding %d nodes from %d section documents...",
        len(nodes), len(documents),
    )
    index = VectorStoreIndex(nodes, show_progress=True)
    target.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(target))
    logger.info("Index persisted: %d nodes -> %s", len(nodes), target)
    return index


def load_index(cfg: IndexConfig | None = None) -> VectorStoreIndex:
    """Load a previously-persisted index. Raises if missing."""
    cfg = cfg or IndexConfig()
    configure_settings(llm=None)
    target = cfg.storage_dir()
    if not target.exists():
        raise FileNotFoundError(
            f"No persisted index at {target}. "
            f"Run `uv run python scripts/build_index.py` first."
        )
    storage_context = StorageContext.from_defaults(persist_dir=str(target))
    return load_index_from_storage(storage_context)


def verify_reload(cfg: IndexConfig | None = None) -> bool:
    """Sanity check: build (or load) then immediately reload and confirm
    that node count matches. Used by ``build_index.py`` as a smoke test.
    """
    cfg = cfg or IndexConfig()
    idx_a = build_index(cfg)
    idx_b = load_index(cfg)
    n_a = len(idx_a.docstore.docs)
    n_b = len(idx_b.docstore.docs)
    if n_a != n_b:
        raise RuntimeError(
            f"Reload mismatch: built={n_a} nodes, reloaded={n_b} nodes."
        )
    logger.info("Reload verified: %d nodes round-trip cleanly.", n_a)
    return True
