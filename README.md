# LLMOps Observability Lab

Production-grade LLMOps stack: observability (Langfuse), reliability patterns, RAG with empirical retrieval evaluation. Built on Anthropic Claude, Qdrant, and Voyage AI.

This repository is a hands-on lab for building production-ready LLM systems from first principles — no LangChain abstractions, every component implemented and tested directly against vendor APIs.

---

## Highlights

- **35+ pytest tests** across reliability, retrieval, and metrics modules
- **mypy strict** across the entire codebase
- **4 retrieval strategies empirically compared** with hand-labeled ground truth
- **Self-hosted observability stack** (Langfuse + Qdrant) via Docker Compose
- **88% input-token cost reduction** demonstrated via prompt caching

See [`day08-observability/lf-client/EVAL_RESULTS.md`](day08-observability/lf-client/EVAL_RESULTS.md) for the full evaluation report with numbers.

---

## Tech stack

| Layer | Tool |
|---|---|
| LLM | Anthropic Claude (Sonnet 4.5, Haiku 4.5) |
| Embeddings | Voyage AI `voyage-3.5` (1024-dim) |
| Reranker | Voyage AI `rerank-2` |
| Vector DB | Qdrant (self-hosted, HNSW) |
| Sparse retrieval | BM25 (`rank-bm25`) |
| Observability | Langfuse 4.x (self-hosted) + OpenTelemetry |
| Resilience | `tenacity` (retry), custom circuit breaker, multi-provider fallback |
| Package manager | `uv` |
| Type checking | `mypy --strict` |
| Linting | `ruff` |
| Testing | `pytest` + `pytest-asyncio` |

---

## Repository structure

This repo represents a structured LLMOps learning journey. The flagship project lives in `day08-observability/lf-client/` — all other day directories are stepping stones.

```
day01-python/         # Async Python, pydantic, pytest, mypy fundamentals
day03-anthropic/      # Anthropic SDK wrapper with cost tracking
day04-tools/          # Tool use: single tool, multi-tool agentic loop
day05-streaming/      # Streaming responses, cancellation
day06-embeddings/     # Voyage embeddings, cosine similarity, batching
day08-observability/  # ★ Flagship: full production stack
└── lf-client/
    ├── src/lf_client/
    │   ├── client.py            # ClaudeClient: observability + retry + caching
    │   ├── usage.py             # Cost tracking (incl. cache tokens)
    │   ├── circuit_breaker.py   # Custom 3-state circuit breaker
    │   ├── fallback.py          # Multi-provider fallback chain
    │   ├── flaky_client.py      # Fault injection for resilience testing
    │   ├── embedding_service.py # Voyage embeddings with batch & cost tracking
    │   ├── reranker.py          # Voyage rerank-2 wrapper
    │   ├── md_parse.py          # Markdown parser
    │   ├── ipynb_parse.py       # Jupyter notebook parser
    │   ├── md_chunking.py       # Document-aware chunking (header hierarchy)
    │   ├── bm25_index.py        # BM25 sparse retrieval
    │   ├── hybrid_search.py     # Dense + BM25 fusion (RRF)
    │   ├── retrieval.py         # Two-stage retrieval (dense → rerank)
    │   ├── eval_data.py         # Hand-labeled ground truth
    │   ├── eval_metrics.py      # recall@k, precision@k, MRR
    │   └── eval_runner.py       # Strategy comparison runner
    ├── tests/                   # 35+ pytest tests
    ├── docker-compose.yml             # Langfuse stack
    ├── docker-compose.qdrant.yml      # Qdrant
    └── EVAL_RESULTS.md          # Empirical retrieval evaluation report
```

---

## What's inside the flagship project

The `lf-client` package is a production-grade LLM client with four major capability groups, built incrementally.

### 1. Observability

`ClaudeClient` automatically emits Langfuse traces with token counts, cost, latency, metadata, and parent-child spans for multi-step workflows. OpenTelemetry-compatible: traces use OTel under the hood (Langfuse 4.x is built on OTel).

Multi-step workflows (RAG, agents) produce nested spans — embed call → retrieval span → generation span → tool calls — all linked under a single trace.

### 2. Reliability

Three layers of failure handling, each addressing a different class of failures:

| Failure type | Solution |
|---|---|
| Transient (429, 529, 5xx) | **Retry** with exponential backoff + jitter via `tenacity` |
| Sustained outage | **Circuit breaker** (3-state: CLOSED / OPEN / HALF_OPEN) |
| Provider-level failure | **Fallback chain** (Claude Sonnet → Haiku → OpenAI) |

A custom `FlakyAsyncAnthropic` drop-in replacement enables **fault injection** — the resilience layer is tested by injecting controlled failures, not just hoping it works.

### 3. Cost engineering

`ClaudeClient` integrates **Anthropic prompt caching** via `cache_control={"type": "ephemeral"}` on system prompts. Empirically validated: large system prompts (~4000 tokens) showed 88% input-token cost reduction on repeated requests. `calculate_cost` accounts separately for cache_read, cache_write, input, and output token pricing.

### 4. RAG ingestion and retrieval

End-to-end pipeline for heterogeneous inputs (markdown + Jupyter notebooks):

```
.md / .ipynb → parse → clean → document-aware chunking → 
embed (Voyage) → index (Qdrant + BM25)
```

**Chunking strategy:** split by markdown headers (H1/H2/H3) first; recursively split oversized sections by token count. Each chunk carries `section_path` metadata (e.g. `"Orchestrator-Workers Workflow > How It Works"`) for citation and filtering.

**Retrieval strategies implemented and compared:**
1. Dense-only (Voyage embeddings + Qdrant HNSW)
2. Dense + reranker (two-stage: top-20 candidates → rerank to top-3)
3. Hybrid (dense + BM25 via Reciprocal Rank Fusion)
4. Hybrid + reranker

See `EVAL_RESULTS.md` for measured performance differences.

---

## Quick start

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager
- Docker + Docker Compose
- Anthropic API key, Voyage API key
- (Optional) OpenAI API key for the fallback chain demo

### Setup

```bash
git clone https://github.com/kupolua/llmops-observability-lab.git
cd llmops-observability-lab/day08-observability/lf-client

# Install dependencies
uv sync

# Copy env template and add your keys
cp .env.example .env  # then edit .env

# Start the observability stack
docker compose up -d                          # Langfuse
docker compose -f docker-compose.qdrant.yml up -d  # Qdrant

# Run the test suite
uv run pytest -v

# Try the demos
uv run python -m lf_client.resilient_demo     # retry + fault injection
uv run python -m lf_client.cb_demo            # circuit breaker
uv run python -m lf_client.caching_demo       # prompt caching savings
```

### Endpoints

- Langfuse UI: http://localhost:3000
- Qdrant dashboard: http://localhost:6333/dashboard

### Quality checks

```bash
uv run ruff format .
uv run ruff check .
uv run mypy
uv run pytest -v
```

All four pass on every commit to the flagship project.

---

## Engineering principles applied

- **Dependency injection over global state** — `ClaudeClient` takes its Anthropic client as a constructor parameter for testability.
- **Fault injection for resilience testing** — `FlakyAsyncAnthropic` simulates 429/529/5xx errors with configurable failure rate.
- **Measure before deciding** — retrieval strategy choices are backed by recall@k, precision@k, and MRR on hand-labeled ground truth, not vibes.
- **mypy strict everywhere** — every public function has type annotations; `Any` is avoided except where unavoidable (untyped third-party SDKs).
- **Costs are first-class** — every API call returns token counts and dollar cost. Total spend per session is exposed as a property.

---

## Limitations and next steps

This is an active project; current limitations are documented honestly:

- **Evaluation sample size is small** (5 queries). Production-grade evaluation requires 50-200 queries with multi-annotator agreement.
- **Single corpus tested** (Anthropic cookbook). Findings about retrieval strategies (e.g. reranker did not improve metrics) may not generalize.
- **No multi-query retrieval yet** for compositional queries — the evaluation shows recall drops to 0.5 on "X vs Y" type questions.
- **No production deployment** — Docker Compose only; no Kubernetes, no horizontal scaling.

Planned next steps:
- Multi-query retrieval for compositional queries
- Query router (rule-based or LLM-based) to choose retrieval strategy per query type
- Grounded generation with citation and confidence threshold (refuse-to-answer for unanswerable queries)
- Larger evaluation dataset

---

## Author

Full-stack developer (Node.js + React) and DevOps Engineer with 15 years of experience, transitioning into LLMOps. Based in Ukraine. Open to LLMOps and AI Infrastructure roles.

- GitHub: [@kupolua](https://github.com/kupolua)

---

## License

This is a learning lab. The code is provided as-is for educational reference.
