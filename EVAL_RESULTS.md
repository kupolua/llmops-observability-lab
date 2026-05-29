# RAG Retrieval Evaluation Results

Empirical comparison of 4 retrieval strategies on a technical documentation corpus.
Hand-labeled ground truth, custom metrics implementation.

---

## TL;DR

| Strategy | Recall@3 | Precision@3 | MRR |
|---|---|---|---|
| **Dense only** | **0.750** | **0.500** | **1.000** |
| Dense + reranker | 0.750 | 0.417 | 0.625 |
| Hybrid (RRF) | 0.750 | 0.500 | 0.875 |
| Hybrid + reranker | 0.750 | 0.417 | 0.625 |

**Counterintuitive finding:** on this corpus, the simplest strategy (dense-only) matches or beats all alternatives across every metric. The reranker actively hurt precision and MRR. This justifies removing the reranker from the production pipeline for this corpus — saving cost and latency without sacrificing quality.

This is **a specific finding for a specific corpus**, not a universal claim. The same evaluation methodology applied to a different corpus might justify the opposite decision.

---

## Setup

### Corpus

A curated subset of the [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook):

- **Documents:** 15 files (`.md` + `.ipynb`) from `patterns/agents/`, `tool_use/`, `observability/`, and repository root
- **Chunks:** 154 after document-aware chunking by markdown header hierarchy
- **Embedding model:** Voyage AI `voyage-3.5` (1024 dimensions, cosine similarity)
- **Vector store:** Qdrant (HNSW index, self-hosted via Docker)
- **Sparse index:** BM25 (`rank-bm25`)

### Ground truth

5 queries hand-labeled by inspecting the source files. Each query is mapped to zero or more (file, section) markers. A chunk is "relevant" if its source contains the file suffix AND its section path starts with the marker prefix.

```python
EVAL_DATASET = [
    EvalQuery(
        query="How does prompt chaining work?",
        relevant=[
            ("basic_workflows.ipynb", "Basic Multi-LLM Workflows"),
        ],
    ),
    EvalQuery(
        query="When should I use parallel tool calls?",
        is_unanswerable=True,
        note="Cookbook describes HOW to use parallel tools, not WHEN",
    ),
    EvalQuery(
        query="Difference between evaluator-optimizer and orchestrator-workers?",
        relevant=[
            ("orchestrator_workers.ipynb", "Orchestrator-Workers Workflow"),
            ("evaluator_optimizer.ipynb", "Evaluator-Optimizer Workflow"),
        ],
        note="Compositional — requires BOTH chunks for a complete answer",
    ),
    EvalQuery(
        query="How to validate JSON output from Claude?",
        relevant=[
            ("extracting_structured_json.ipynb", "Extracting Structured JSON using Claude and Tool Use"),
        ],
    ),
    EvalQuery(
        query="What is tool_choice and when to use force?",
        relevant=[
            ("tool_choice.ipynb", "Tool choice"),
            ("tool_choice.ipynb", "Forcing a specific tool"),
        ],
    ),
]
```

The dataset is deliberately heterogeneous:
- **Q1, Q4** — single-document, single-section (simple)
- **Q2** — unanswerable from corpus (tests refuse-to-answer behavior)
- **Q3** — compositional, requires retrieving 2 separate documents
- **Q5** — single-document but requires 2 sections from it

### Strategies compared

1. **Dense only** — Voyage embeddings → Qdrant cosine similarity → top-3
2. **Dense + reranker** — top-20 from Qdrant → Voyage `rerank-2` → top-3
3. **Hybrid (RRF)** — top-20 from dense + top-20 from BM25 → Reciprocal Rank Fusion (k=60) → top-3
4. **Hybrid + reranker** — same as #3 but reranker reorders the top-20 fused candidates

### Metrics

- **Recall@k** — fraction of ground truth markers that have at least one matching chunk in top-k. *(Note: counted per-marker, not per-chunk, to avoid the over-counting bug where multiple chunks matching one marker would inflate recall above 1.0.)*
- **Precision@k** — fraction of top-k chunks that are relevant to any ground truth marker
- **MRR** — 1 / (rank of first relevant chunk in top-k); 0 if no relevant chunk found
- Unanswerable queries are excluded from averages (handled separately — see Q2 discussion below)

---

## Per-query breakdown

### Q1: "How does prompt chaining work?"

Ground truth: 1 relevant chunk in `basic_workflows.ipynb`.

| Strategy | Recall | Precision | First-relevant rank |
|---|---|---|---|
| Dense only | 1.00 | 0.33 | **1** |
| Dense + reranker | 1.00 | 0.33 | 2 |
| Hybrid (RRF) | 1.00 | 0.33 | **1** |
| Hybrid + reranker | 1.00 | 0.33 | 2 |

**Observation:** Dense found the correct chunk at rank 1. The reranker pushed it down to rank 2, surfacing a less relevant chunk from `tool_search_alternate_approaches.ipynb` at rank 1 — likely because that document also contains multi-step examples that look like "chains" to the cross-encoder.

### Q2: "When should I use parallel tool calls?" (UNANSWERABLE)

The corpus describes **how** to use parallel tool calls but doesn't contain a "when to use" section. This is included deliberately to test the system's behavior on unanswerable queries.

All four strategies confidently returned chunks anyway — including the reranker, which assigned a confident relevance score (0.738) to a chunk that doesn't actually answer the question.

**Implication:** high reranker confidence does not mean the answer is correct — it only means the chunk is the closest match among candidates. A production system needs a confidence threshold mechanism that recognizes "no good answer exists" and refuses to answer rather than presenting the closest mismatch.

### Q3: "Difference between evaluator-optimizer and orchestrator-workers?" (COMPOSITIONAL)

Ground truth: 2 relevant chunks (one per workflow).

| Strategy | Recall | Precision | First-relevant rank |
|---|---|---|---|
| Dense only | 0.50 | 0.67 | **1** |
| Dense + reranker | 0.50 | 0.33 | 2 |
| Hybrid (RRF) | 0.50 | 0.67 | 2 |
| Hybrid + reranker | 0.50 | 0.33 | 2 |

**Observation:** All strategies find only the orchestrator-workers chunk — the evaluator-optimizer chunk doesn't surface in top-3 for any strategy. This is the **fundamental limitation of single-pass retrieval** for compositional queries.

**Implication:** Compositional queries need a different architecture — multi-query retrieval (decompose into sub-queries, retrieve separately, merge) or an agentic approach (LLM decides what to search for). No amount of reranking fixes this if the second relevant chunk doesn't reach the top-k candidate pool.

### Q4: "How to validate JSON output from Claude?"

Ground truth: 1 relevant chunk in `extracting_structured_json.ipynb`.

| Strategy | Recall | Precision | First-relevant rank |
|---|---|---|---|
| Dense only | 1.00 | 0.67 | **1** |
| Dense + reranker | 1.00 | 0.67 | 2 |
| Hybrid (RRF) | 1.00 | 0.67 | **1** |
| Hybrid + reranker | 1.00 | 0.67 | 2 |

**Observation:** Dense finds the right intro chunk at rank 1. Reranker pushes the README's table-of-contents to rank 1 — technically the README has a link to JSON validation, but it's a navigation page, not an answer. This is a **subtle precision problem** that the rank-only metric doesn't fully capture.

### Q5: "What is tool_choice and when to use force?"

Ground truth: 2 relevant chunks in `tool_choice.ipynb` (intro + forcing section).

| Strategy | Recall | Precision | First-relevant rank |
|---|---|---|---|
| Dense only | 0.50 | 0.33 | **1** |
| Dense + reranker | 0.50 | 0.33 | **1** |
| Hybrid (RRF) | 0.50 | 0.33 | **1** |
| Hybrid + reranker | 0.50 | 0.33 | **1** |

**Observation:** All strategies find the intro section but miss the "Forcing a specific tool" subsection within the same file. The chunker grouped the intro into one chunk and the forcing example into a separate chunk; their embeddings differ enough that only one is retrieved. This is a **chunking problem**, not a retrieval problem — even ideal retrieval can't recover what chunking separated.

---

## Key findings

### 1. Reranker did not help — it actively hurt

Across all four metrics where the reranker differed from dense, it was equal or worse:

- **Recall@3** — identical (0.750). Reranker didn't find anything dense missed.
- **Precision@3** — worse (0.500 → 0.417). Reranker introduced noise.
- **MRR** — much worse (1.000 → 0.625). Reranker pushed correct answers off rank 1.

This is **opposite to the common assumption** that adding a reranker always improves retrieval. Likely reasons specific to this corpus:

- Voyage `rerank-2` is trained on broad data, not on Anthropic's specific documentation style
- The cross-encoder picks up "semantic vicinity" even when the user wanted a specific term
- The corpus is small (154 chunks) and topically focused — dense embeddings already discriminate well

### 2. BM25 + dense (hybrid) showed no measurable improvement either

Identical to dense-only across recall@3 and precision@3, slightly worse on MRR (0.875 vs 1.000). BM25's keyword-matching strength didn't pay off because none of the test queries hinged on rare terms only BM25 could catch.

This validates: **hybrid search is not free**. It adds latency and code complexity. Use it only when the corpus or query distribution has properties where it's expected to help.

### 3. Recall@3 = 0.75 is a ceiling caused by compositional queries

The 0.75 average is the arithmetic mean across 4 answerable queries: `(1.0 + 0.5 + 1.0 + 0.5) / 4`. The two `0.5`s come from Q3 and Q5, which both need 2 relevant chunks but only get 1.

**No single-pass retrieval strategy fixes this.** Multi-query retrieval or agentic search is required.

### 4. Confidence scores are misleading on unanswerable queries

Q2 ("When should I use parallel tool calls?") has no answer in the corpus, yet the reranker confidently scored its top result at 0.738 — higher than several correctly-answered queries. **Confidence reflects closeness within the candidate pool, not absolute correctness.**

A production system must layer additional logic on top: e.g., "if no chunk reaches a rerank threshold AND the LLM is forced to answer from a confidence floor, refuse to answer."

---

## Production decision

For **this specific corpus and query distribution**, the recommended retrieval configuration is:

```
Dense-only retrieval (Voyage voyage-3.5 + Qdrant HNSW, top-k=3)
```

**Rationale:**
- Best or tied-best on all three metrics
- ~$0.05 per million tokens saved by skipping the reranker
- Lower latency (no second model call)
- Simpler operationally (one fewer dependency)

The reranker code remains in the repo and is testable — if the corpus changes (e.g., gets larger or more topically diverse), re-running this evaluation may flip the decision.

---

## Limitations

This evaluation is honest about what it does and doesn't establish:

| Limitation | Implication |
|---|---|
| n=5 queries | Wide confidence interval; not statistically robust |
| Single annotator (project author) | Ground truth has inherent subjectivity |
| Single corpus (Anthropic cookbook) | Findings may not generalize |
| English-only queries | Multilingual behavior untested |
| No latency benchmarks included | Operational trade-offs partially unmeasured |
| No measurement of "refuse-to-answer" correctness on Q2 | Unanswerable handling discussed qualitatively only |

**Production-grade evaluation requires** at minimum: 50-200 queries, multi-annotator agreement (Cohen's kappa or similar), and stratified sampling across query types.

---

## Reproducing these results

```bash
cd day08-observability/lf-client

# Start Qdrant
docker compose -f docker-compose.qdrant.yml up -d

# Index the corpus (clones anthropic-cookbook on first run)
uv run python -m lf_client.index_cookbook

# Run the full evaluation
uv run python -m lf_client.eval_runner

# Run metric unit tests
uv run pytest tests/test_eval_metrics.py -v
```

Expected runtime: ~30 seconds for evaluation, ~1 second for unit tests. Expected Voyage API cost: under $0.01 per full evaluation run.

---

## Next steps

In priority order:

1. **Multi-query retrieval for compositional queries** — decompose "X vs Y" questions into separate retrievals, then merge. Should raise Q3 recall from 0.5 toward 1.0.
2. **Query router** — rule-based or LLM-based classification of incoming queries into types (compositional / exact-term / conceptual), routing each to the optimal strategy.
3. **Grounded generation with confidence threshold** — wire retrieval into Claude with explicit "refuse to answer if no relevant chunk above threshold" logic.
4. **Larger evaluation set** — 50+ queries across categories, with second annotator for agreement measurement.
5. **Latency benchmarks** — add p50/p95 latency per strategy alongside quality metrics.
