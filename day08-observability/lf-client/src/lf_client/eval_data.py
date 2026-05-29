"""Ground truth для оценки RAG retrieval.

Размечено вручную в день 17. Каждый запрос имеет:
- список (filename_suffix, section_prefix) релевантных чанков
- или пустой список = unanswerable (корпус не покрывает)
"""

from dataclasses import dataclass, field


@dataclass
class GroundTruthRelevant:
    """Маркер релевантного чанка.

    file_suffix: подстрока source path. Например, "basic_workflows.ipynb".
    section_prefix: префикс section_path. Например, "Orchestrator-Workers Workflow".
    """

    file_suffix: str
    section_prefix: str


@dataclass
class EvalQuery:
    query: str
    relevant: list[GroundTruthRelevant] = field(default_factory=list)
    is_unanswerable: bool = False
    note: str = ""


EVAL_DATASET: list[EvalQuery] = [
    EvalQuery(
        query="How does prompt chaining work?",
        relevant=[
            GroundTruthRelevant(
                file_suffix="basic_workflows.ipynb",
                section_prefix="Basic Multi-LLM Workflows",
            ),
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
            GroundTruthRelevant(
                file_suffix="orchestrator_workers.ipynb",
                section_prefix="Orchestrator-Workers Workflow",
            ),
            GroundTruthRelevant(
                file_suffix="evaluator_optimizer.ipynb",
                section_prefix="Evaluator-Optimizer Workflow",
            ),
        ],
        note="Compositional query — needs BOTH chunks",
    ),
    EvalQuery(
        query="How to validate JSON output from Claude?",
        relevant=[
            GroundTruthRelevant(
                file_suffix="extracting_structured_json.ipynb",
                section_prefix="Extracting Structured JSON using Claude and Tool Use",
            ),
        ],
    ),
    EvalQuery(
        query="What is tool_choice and when to use force?",
        relevant=[
            GroundTruthRelevant(
                file_suffix="tool_choice.ipynb",
                section_prefix="Tool choice",
            ),
            GroundTruthRelevant(
                file_suffix="tool_choice.ipynb",
                section_prefix="Forcing a specific tool",
            ),
        ],
    ),
]
