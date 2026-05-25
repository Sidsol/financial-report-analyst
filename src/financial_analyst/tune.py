"""Tuning sweep for retrieval parameters.

Builds a capped grid of indexes (varying chunk_size × overlap), then sweeps
retrieval-time parameters (top_k × use_filters × use_reranker) against the
fixed eval set. Indexes are cached on disk and reused across sweeps.

Results are persisted as both JSON and a markdown comparison table. The
table compares each tuned config against the Lab 7.2 baseline.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import pandas as pd

from .evaluate import EvalSummary, evaluate, load_questions
from .index import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, IndexConfig, build_index
from .paths import RESULTS_DIR

logger = logging.getLogger(__name__)


@dataclass
class TuningConfig:
    """One row of the sweep."""

    chunk_size: int
    chunk_overlap: int
    top_k: int
    use_filters: bool
    use_reranker: bool

    @property
    def name(self) -> str:
        rerank_tag = "_rr" if self.use_reranker else ""
        filt_tag = "" if self.use_filters else "_nofilt"
        return f"c{self.chunk_size}_o{self.chunk_overlap}_k{self.top_k}{filt_tag}{rerank_tag}"

    def index_config(self) -> IndexConfig:
        return IndexConfig(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )


@dataclass
class TuningRow:
    config: TuningConfig
    metadata_hit_rate: float
    strict_hit_rate: float
    metadata_mrr: float
    strict_mrr: float
    eval_seconds: float
    n_nodes: int


def _default_sweep() -> list[TuningConfig]:
    """Capped grid recommended by the project plan.

    * 2 chunk sizes × 2 overlaps = 4 indexes to build (cached)
    * 2 top-k values × 2 filter settings × 2 reranker settings = 8 runtime
      configs per index
    * Reranker is applied only to the best embedding config (decided after
      the no-rerank pass), keeping the total wallclock manageable.
    """
    configs: list[TuningConfig] = []
    for chunk_size, overlap in product([512, 1024], [50, 100]):
        for top_k in (5, 10):
            for use_filters in (True, False):
                configs.append(TuningConfig(
                    chunk_size=chunk_size, chunk_overlap=overlap,
                    top_k=top_k, use_filters=use_filters, use_reranker=False,
                ))
    return configs


def _rerank_sweep(best: TuningConfig) -> list[TuningConfig]:
    """Add reranker on top of the best non-rerank config."""
    return [
        TuningConfig(
            chunk_size=best.chunk_size, chunk_overlap=best.chunk_overlap,
            top_k=top_k, use_filters=True, use_reranker=True,
        )
        for top_k in (5, 10)
    ]


def run_sweep(configs: list[TuningConfig] | None = None) -> list[TuningRow]:
    """Run the full sweep, building indexes lazily and reusing them."""
    configs = configs or _default_sweep()
    # Group by IndexConfig so we only load each index once.
    questions = load_questions()
    rows: list[TuningRow] = []

    indexes: dict[tuple[int, int], object] = {}
    nodes_count: dict[tuple[int, int], int] = {}

    for cfg in configs:
        key = (cfg.chunk_size, cfg.chunk_overlap)
        if key not in indexes:
            ic = cfg.index_config()
            logger.info(
                "Building/loading index for chunk_size=%d overlap=%d ...",
                cfg.chunk_size, cfg.chunk_overlap,
            )
            idx = build_index(ic)
            indexes[key] = idx
            nodes_count[key] = len(idx.docstore.docs)
        idx = indexes[key]

        t0 = time.perf_counter()
        summary = evaluate(
            idx,
            top_k=cfg.top_k,
            use_filters=cfg.use_filters,
            use_reranker=cfg.use_reranker,
            candidate_top_k=max(cfg.top_k * 4, 20),
            questions=questions,
            config_name=cfg.name,
        )
        dt = time.perf_counter() - t0
        logger.info(
            "  %s -> md@%d=%.3f strict@%d=%.3f md_mrr=%.3f strict_mrr=%.3f (%.1fs)",
            cfg.name, cfg.top_k, summary.metadata_hit_rate,
            cfg.top_k, summary.strict_hit_rate,
            summary.metadata_mrr, summary.strict_mrr, dt,
        )
        rows.append(TuningRow(
            config=cfg,
            metadata_hit_rate=summary.metadata_hit_rate,
            strict_hit_rate=summary.strict_hit_rate,
            metadata_mrr=summary.metadata_mrr,
            strict_mrr=summary.strict_mrr,
            eval_seconds=dt,
            n_nodes=nodes_count[key],
        ))
    return rows


def run_full_sweep_with_rerank() -> list[TuningRow]:
    """Run the default sweep, then add a reranker on the best config."""
    rows = run_sweep()
    # Pick the best non-rerank config by strict MRR.
    best = max(rows, key=lambda r: r.strict_mrr).config
    logger.info(
        "Best non-rerank config: %s. Adding reranker variants.", best.name
    )
    rows.extend(run_sweep(_rerank_sweep(best)))
    return rows


def rows_to_dataframe(rows: list[TuningRow]) -> pd.DataFrame:
    return pd.DataFrame([{
        "config": r.config.name,
        "chunk_size": r.config.chunk_size,
        "overlap": r.config.chunk_overlap,
        "top_k": r.config.top_k,
        "filters": r.config.use_filters,
        "rerank": r.config.use_reranker,
        "nodes": r.n_nodes,
        "metadata_hit": r.metadata_hit_rate,
        "strict_hit": r.strict_hit_rate,
        "metadata_mrr": r.metadata_mrr,
        "strict_mrr": r.strict_mrr,
        "seconds": r.eval_seconds,
    } for r in rows])


def write_tuning_outputs(rows: list[TuningRow], out_dir: Path | None = None) -> tuple[Path, Path]:
    """Persist JSON and a markdown comparison table."""
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df = rows_to_dataframe(rows).sort_values("strict_mrr", ascending=False)

    json_path = out_dir / "tuning.json"
    json_path.write_text(
        json.dumps([{
            "config": asdict(r.config),
            "metadata_hit_rate": r.metadata_hit_rate,
            "strict_hit_rate": r.strict_hit_rate,
            "metadata_mrr": r.metadata_mrr,
            "strict_mrr": r.strict_mrr,
            "eval_seconds": r.eval_seconds,
            "n_nodes": r.n_nodes,
        } for r in rows], indent=2),
        encoding="utf-8",
    )

    # Identify baseline (Lab 7.2 defaults: c=512, o=50, k=5, filters, no rerank).
    baseline_mask = (
        (df["chunk_size"] == DEFAULT_CHUNK_SIZE)
        & (df["overlap"] == DEFAULT_CHUNK_OVERLAP)
        & (df["top_k"] == 5)
        & (df["filters"] == True)  # noqa: E712
        & (df["rerank"] == False)  # noqa: E712
    )
    if baseline_mask.any():
        b = df[baseline_mask].iloc[0]
        baseline_md = b["metadata_hit"]
        baseline_st = b["strict_hit"]
        baseline_md_mrr = b["metadata_mrr"]
        baseline_st_mrr = b["strict_mrr"]
    else:
        baseline_md = baseline_st = baseline_md_mrr = baseline_st_mrr = float("nan")

    def delta(x: float, base: float) -> str:
        if pd.isna(base):
            return ""
        diff = x - base
        sign = "+" if diff >= 0 else ""
        return f" ({sign}{diff:.3f})"

    lines = [
        "# Retrieval Tuning — Comparison Table",
        "",
        "Sorted by **strict MRR** (descending). Baseline = "
        f"chunk={DEFAULT_CHUNK_SIZE}, overlap={DEFAULT_CHUNK_OVERLAP}, top_k=5, filters=on, rerank=off.",
        "",
        "| Config | chunk | ovlp | k | filt | rerank | nodes | Md-Hit | Strict-Hit | Md-MRR | Strict-MRR | sec |",
        "|---|---:|---:|---:|:-:|:-:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        is_baseline = (
            row["chunk_size"] == DEFAULT_CHUNK_SIZE
            and row["overlap"] == DEFAULT_CHUNK_OVERLAP
            and row["top_k"] == 5
            and bool(row["filters"])
            and not bool(row["rerank"])
        )
        marker = "**B**" if is_baseline else ""
        lines.append(
            f"| {row['config']} {marker} | {row['chunk_size']} | {row['overlap']} | "
            f"{row['top_k']} | {'Y' if row['filters'] else '·'} | {'Y' if row['rerank'] else '·'} | "
            f"{row['nodes']} | {row['metadata_hit']:.3f}{delta(row['metadata_hit'], baseline_md)} | "
            f"{row['strict_hit']:.3f}{delta(row['strict_hit'], baseline_st)} | "
            f"{row['metadata_mrr']:.3f}{delta(row['metadata_mrr'], baseline_md_mrr)} | "
            f"{row['strict_mrr']:.3f}{delta(row['strict_mrr'], baseline_st_mrr)} | "
            f"{row['seconds']:.1f} |"
        )
    md_path = out_dir / "tuning.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
