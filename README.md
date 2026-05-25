# Financial Report Analyst

A LlamaIndex RAG (Retrieval-Augmented Generation) assistant that answers
questions over public-company 10-K filings, built as a three-part class
project covering:

- **Lab 7.1** — Build a reproducible RAG corpus
- **Lab 7.2** — Retrieval evaluation (Hit Rate + MRR)
- **Lab 7.3** — Retrieval tuning (chunking, metadata filters, reranking)

## Corpus

Last three annual 10-K filings of three mega-cap technology companies,
downloaded directly from [SEC EDGAR](https://www.sec.gov/edgar):

| Company   | Ticker | CIK        |
|-----------|--------|------------|
| Microsoft | MSFT   | 0000789019 |
| Apple     | AAPL   | 0000320193 |
| NVIDIA    | NVDA   | 0001045810 |

The exact set of filings is pinned in [`data/filings_manifest.yaml`](data/filings_manifest.yaml).

## Stack

| Concern        | Choice                                                    |
|----------------|-----------------------------------------------------------|
| Pkg manager    | [`uv`](https://docs.astral.sh/uv/) (Python 3.11)          |
| Framework      | [LlamaIndex](https://www.llamaindex.ai/) core             |
| Embeddings     | HuggingFace `BAAI/bge-small-en-v1.5` (local)              |
| LLM            | OpenAI **or** GitHub Models (switch via `LLM_PROVIDER`)   |
| Vector store   | LlamaIndex `SimpleVectorStore` (persisted to `storage/`)  |
| Reranker       | `BAAI/bge-reranker-base` (Lab 7.3 only)                   |

## Quick start

```powershell
# 1. Install dependencies (creates .venv)
uv sync --all-extras

# 2. Configure environment
Copy-Item .env.example .env
# Edit .env to set GITHUB_TOKEN (or OPENAI_API_KEY) and confirm SEC_USER_AGENT.

# 3. Download the 10-K filings (idempotent — re-runs are cheap)
uv run python scripts/download_reports.py

# 4. Build the vector index
uv run python scripts/build_index.py

# 5. Ask a question
uv run python scripts/run_query.py "How did NVIDIA's data center revenue change from FY2023 to FY2024?"

# 6. Run the retrieval evaluation
uv run python scripts/run_eval.py

# 7. Run the tuning sweep
uv run python scripts/run_tuning.py
```

## Trying it out

Once the steps above have run once, the index and indexes-for-tuning are
cached on disk and these commands are fast.

### Ask a question (simple query engine)
```powershell
uv run python scripts/run_query.py "How did NVIDIA's data center revenue change from FY2023 to FY2024?"
uv run python scripts/run_query.py --top-k 10 "Compare R&D spending across MSFT, AAPL, and NVDA"
uv run python scripts/run_query.py --no-filters "What did management say about AI demand?"
uv run python scripts/run_query.py --verbose "Compare Apple's and Microsoft's FY2024 risk disclosures"
```

### Ask via the AgentWorkflow (Lab 7.3)
The agent runs `classify_query → retrieve → validate → synthesize → cite → log_failure_if_low_confidence`.
```powershell
uv run python scripts/run_query.py --agent --top-k 10 "What climate risks do these companies identify?"
uv run python scripts/run_query.py --agent --no-filters "Recipe for chocolate chip cookies"  # triggers low-confidence log
Get-Content results\agent_failures.jsonl
```

### Reproduce the eval (Lab 7.2)
```powershell
uv run python scripts/run_eval.py --top-k 5                              # baseline
uv run python scripts/run_eval.py --top-k 10 --config-name k10           # alternate
Get-Content results\eval_baseline.md
Get-Content results\failures_baseline.md
```

### Reproduce the tuning sweep (Lab 7.3)
```powershell
uv run python scripts/run_tuning.py --quick    # no reranker, ~5s after caches warm
uv run python scripts/run_tuning.py            # full sweep with reranker (~5 min)
Get-Content results\tuning.md
```

### Swap LLM provider
Edit `.env`:
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```
Then re-run any query — embeddings stay local, only synthesis changes.

### Force a fresh index build at a custom chunk size
```powershell
uv run python scripts/build_index.py --force --chunk-size 1024 --chunk-overlap 50
```

### Run a notebook end-to-end
```powershell
uv run jupyter notebook notebooks\lab_7_1_build_rag.ipynb
```
Or open in VS Code: install the Python + Jupyter extensions, then
`Ctrl+Shift+P → Python: Select Interpreter → .venv\Scripts\python.exe`.

## Notebooks

The three lab deliverables also live as notebooks under `notebooks/`:

- `notebooks/lab_7_1_build_rag.ipynb`
- `notebooks/lab_7_2_retrieval_eval.ipynb`
- `notebooks/lab_7_3_retrieval_tuning.ipynb`

They import from `src/financial_analyst/` rather than duplicating logic, so
running the scripts above is enough to reproduce all notebook outputs.

## Lab reports

Written deliverables are in [`reports/`](reports/):

- `reports/lab_7_1.md` — corpus & pipeline reproducibility steps
- `reports/lab_7_2.md` — evaluation results + failure modes
- `reports/lab_7_3.md` — tuning experiments + comparison table

## Repository layout

```
financial-report-analyst/
├── pyproject.toml
├── data/
│   ├── filings_manifest.yaml      # pinned SEC filing accession numbers
│   ├── eval/questions.yaml        # 25 evaluation questions
│   └── raw/                       # downloaded HTML (gitignored)
├── storage/                       # persisted index (gitignored)
├── results/                       # eval + tuning outputs (gitignored)
├── src/financial_analyst/         # the pipeline package
├── scripts/                       # CLI entrypoints
├── notebooks/                     # per-lab walkthroughs
└── reports/                       # written lab reports
```
