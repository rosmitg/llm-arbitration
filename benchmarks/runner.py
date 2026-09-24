# benchmarks/runner.py

import csv
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel

from app.agents.critics import call_model
from app.models import ModelVersion
from benchmarks.loader import (
    BenchmarkExample,
    EvaluationFormat,
    GoldLabel,
    TaskType,
    load_all_benchmarks,
)


# ─────────────────────────────────────────
# Result schema
# ─────────────────────────────────────────


class JudgeResult(BaseModel):
    """
    The result of one judge evaluating one benchmark example.

    Atomic unit of the experiment.
    Every conclusion in metrics.py, calibration.py,
    and correlation.py is derived from these records.

    One row per (example, judge) pair in the CSV.
    Never recompute — load from CSV instead.
    """

    # Which example was evaluated
    example_id: str
    task_type: TaskType
    source: str
    subset: Optional[str] = None

    # Which judge produced this result
    judge: str      # provider: openai / anthropic / google
    model: str      # exact model string

    # Experiment result
    gold_label: GoldLabel
    prediction: GoldLabel
    correct: bool

    # Cost and performance
    latency_ms: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float

    # Raw response for debugging only
    # Never use in metrics calculations
    raw_response: str


# ─────────────────────────────────────────
# Cost table (USD per 1k tokens)
# ─────────────────────────────────────────

COST_PER_1K = {
    ModelVersion.GPT4O: {
        "input": 0.005,
        "output": 0.015,
    },
    ModelVersion.CLAUDE_HAIKU: {
        "input": 0.00025,
        "output": 0.00125,
    },
    ModelVersion.GEMINI_FLASH: {
        "input": 0.000075,
        "output": 0.0003,
    },
}


def estimate_cost(
    model: ModelVersion,
    input_tokens: int,
    output_tokens: int,
) -> float:
    rates = COST_PER_1K.get(model, {"input": 0, "output": 0})
    return (
        input_tokens * rates["input"] / 1000
        + output_tokens * rates["output"] / 1000
    )


# ─────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────


def build_pointwise_prompt(example: BenchmarkExample) -> str:
    """
    Prompt for POINTWISE evaluation.
    Judge sees: instruction + claim + context.
    Judge must answer: PASS or FAIL only.
    """
    return f"""You are an expert evaluator.

TASK:
{example.input}

DOCUMENT (evidence):
{example.context or "No context provided."}

CLAIM TO EVALUATE:
{example.output_a}

Does the document support this claim?

Respond with exactly one word: PASS or FAIL
Do not explain. Do not add any other text.
Your entire response must be either PASS or FAIL."""


def build_pairwise_prompt(example: BenchmarkExample) -> str:
    """
    Prompt for PAIRWISE evaluation.
    Judge sees: instruction + output A + output B.
    Judge must answer: A or B only.
    """
    return f"""You are an expert evaluator.

INSTRUCTION:
{example.input}

OUTPUT A:
{example.output_a}

OUTPUT B:
{example.output_b}

Which output better follows the instruction?

Respond with exactly one letter: A or B
Do not explain. Do not add any other text.
Your entire response must be either A or B."""


# ─────────────────────────────────────────
# Response parser
# ─────────────────────────────────────────


def parse_prediction(
    raw: str,
    evaluation_format: EvaluationFormat,
) -> GoldLabel:
    """
    Parse raw model response into a GoldLabel.
    Models often add extra text despite instructions.
    We extract the label from wherever it appears.
    """
    text = raw.strip().upper()

    if evaluation_format == EvaluationFormat.POINTWISE:
        if "PASS" in text:
            return GoldLabel.PASS
        elif "FAIL" in text:
            return GoldLabel.FAIL
        else:
            raise ValueError(
                f"Could not parse pointwise label from: {raw!r}"
            )

    elif evaluation_format == EvaluationFormat.PAIRWISE:
        if text.startswith("A"):
            return GoldLabel.A
        elif text.startswith("B"):
            return GoldLabel.B
        elif " A " in f" {text} ":
            return GoldLabel.A
        elif " B " in f" {text} ":
            return GoldLabel.B
        else:
            raise ValueError(
                f"Could not parse pairwise label from: {raw!r}"
            )

    else:
        raise ValueError(
            f"Unknown evaluation format: {evaluation_format}"
        )


# ─────────────────────────────────────────
# Core runner
# ─────────────────────────────────────────


def run_one_example(
    example: BenchmarkExample,
    judge: str,
    model: ModelVersion,
) -> JudgeResult:
    """
    Run one judge on one benchmark example.
    Returns a JudgeResult with full metadata.
    """
    if example.evaluation_format == EvaluationFormat.POINTWISE:
        prompt = build_pointwise_prompt(example)
    else:
        prompt = build_pairwise_prompt(example)

    start = time.time()
    raw = call_model(model, prompt)
    latency_ms = (time.time() - start) * 1000

    prediction = parse_prediction(raw, example.evaluation_format)

    input_tokens = len(prompt.split())
    output_tokens = len(raw.split())

    return JudgeResult(
        example_id=example.id,
        task_type=example.task_type,
        source=example.source,
        subset=example.subset,
        judge=judge,
        model=model.value,
        gold_label=example.gold_label,
        prediction=prediction,
        correct=(prediction == example.gold_label),
        latency_ms=round(latency_ms, 2),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=round(
            estimate_cost(model, input_tokens, output_tokens), 6
        ),
        raw_response=raw,
    )


# ─────────────────────────────────────────
# CSV save and load
# ─────────────────────────────────────────

RESULTS_FILE = Path("benchmarks/results/results.csv")

FIELDNAMES = [
    "example_id", "task_type", "source", "subset",
    "judge", "model", "gold_label", "prediction",
    "correct", "latency_ms", "input_tokens",
    "output_tokens", "estimated_cost_usd", "raw_response"
]


def save_result(result: JudgeResult) -> None:
    """
    Append one result row to the CSV file.
    Creates the file with headers if it doesn't exist.
    Called immediately after every API call.
    """
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RESULTS_FILE.exists()

    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result.model_dump())


def load_results() -> list[JudgeResult]:
    """Load all results from CSV."""
    if not RESULTS_FILE.exists():
        return []
    df = pd.read_csv(RESULTS_FILE)
    records = df.to_dict("records")
    # Replace NaN with None for Optional fields
    cleaned = [
        {k: (None if isinstance(v, float) and pd.isna(v) else v)
         for k, v in row.items()}
        for row in records
    ]
    return [JudgeResult(**row) for row in cleaned]


def already_run(example_id: str, judge: str) -> bool:
    """
    Check if this (example, judge) pair is already in the CSV.
    Enables resuming a crashed run without re-paying.
    """
    if not RESULTS_FILE.exists():
        return False
    with open(RESULTS_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row["example_id"] == example_id
                and row["judge"] == judge
            ):
                return True
    return False


# ─────────────────────────────────────────
# Full benchmark run
# ─────────────────────────────────────────

JUDGES = [
    ("anthropic", ModelVersion.CLAUDE_HAIKU),
    ("openai", ModelVersion.GPT4O),
    ("google", ModelVersion.GEMINI_FLASH),
]


def run_benchmark(
    aggrefact_max: int = 500,
    llmbar_max: int = 419,
    judgebench_max: int = 350,
) -> list[JudgeResult]:
    """
    Run all judges on all benchmark examples.
    Saves each result to CSV immediately after the API call.
    Skips already-completed (example, judge) pairs.
    """
    examples = load_all_benchmarks(
        aggrefact_max=aggrefact_max,
        llmbar_max=llmbar_max,
        judgebench_max=judgebench_max,
    )

    total = len(examples) * len(JUDGES)
    done = 0
    skipped = 0
    errors = 0

    print(
        f"\nBenchmark: {len(examples)} examples × "
        f"{len(JUDGES)} judges = {total} calls"
    )
    print(f"Results: {RESULTS_FILE}\n")

    for example in examples:
        for judge, model in JUDGES:

            if already_run(example.id, judge):
                skipped += 1
                done += 1
                continue

            try:
                result = run_one_example(example, judge, model)
                save_result(result)

                status = "✅" if result.correct else "❌"
                print(
                    f"{status} [{judge:10}] {example.id:35} "
                    f"gold={result.gold_label.value:4} "
                    f"pred={result.prediction.value:4} "
                    f"{result.latency_ms:6.0f}ms "
                    f"${result.estimated_cost_usd:.5f}"
                )

            except Exception as e:
                errors += 1
                print(f"⚠️  [{judge}] {example.id} ERROR: {e}")

            done += 1

    print(
        f"\nDone: {done}/{total} | "
        f"Skipped: {skipped} | "
        f"Errors: {errors}"
    )
    return load_results()


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    run_benchmark(
        aggrefact_max=5,
        llmbar_max=5,
        judgebench_max=5,
    )