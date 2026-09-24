# benchmarks/loader.py

from enum import Enum
from typing import Any, Optional

from datasets import load_dataset
from pydantic import BaseModel, Field


# ─────────────────────────────────────────
# Common benchmark schema
# ─────────────────────────────────────────


class TaskType(str, Enum):
    GROUNDEDNESS = "groundedness"
    INSTRUCTION = "instruction"
    REASONING = "reasoning"


class EvaluationFormat(str, Enum):
    """
    POINTWISE:
        Judge one candidate output directly.

        Example:
        Is this claim supported by the document?

    PAIRWISE:
        Compare output A against output B.

        Example:
        Which response better follows the instruction?
    """

    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"


class GoldLabel(str, Enum):
    """
    POINTWISE labels:
        PASS
        FAIL

    PAIRWISE labels:
        A
        B
    """

    PASS = "pass"
    FAIL = "fail"

    A = "a"
    B = "b"


class BenchmarkExample(BaseModel):
    """
    Unified representation used by the rest of Arbiter.

    Every source dataset is converted into this format.

    Pointwise example:
        input
        output_a
        context
        gold_label = PASS / FAIL

    Pairwise example:
        input
        output_a
        output_b
        gold_label = A / B
    """

    id: str

    task_type: TaskType
    evaluation_format: EvaluationFormat

    # Question, instruction, or task
    input: str

    # Candidate being evaluated
    output_a: str

    # Only present for pairwise benchmarks
    output_b: Optional[str] = None

    # Evidence / retrieved document where applicable
    context: Optional[str] = None

    # Ground-truth benchmark answer
    gold_label: GoldLabel

    # Dataset name
    source: str

    # Dataset-specific slice/subset
    subset: Optional[str] = None

    # Preserve useful source metadata for later analysis
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────
# LLM-AggreFact
# ─────────────────────────────────────────

def load_llm_aggrefact(
    max_examples: int = 500,
) -> list[BenchmarkExample]:
    """
    LLM-AggreFact groundedness benchmark.

    Stratified sample across underlying datasets and labels.
    Fixed seed for reproducibility.

    IMPORTANT:
    IDs use the ORIGINAL dataset row index so existing cached
    results remain attached to the correct examples.
    """
    import pandas as pd

    dataset = load_dataset(
        "lytang/LLM-AggreFact",
        split="test",
    )

    df = pd.DataFrame(dataset)

    # Preserve original Hugging Face row index
    df["_source_index"] = df.index

    # Count actual dataset/label groups
    n_groups = df.groupby(["dataset", "label"]).ngroups
    per_group = max(1, max_examples // n_groups)

    sampled_parts = []
    for (_, _), group in df.groupby(["dataset", "label"]):
        sampled_parts.append(
            group.sample(
                n=min(len(group), per_group),
                random_state=42,
            )
        )

    sampled = pd.concat(sampled_parts)

    # Top up if small groups prevented reaching max_examples
    if len(sampled) < max_examples:
        remaining = df[
            ~df["_source_index"].isin(sampled["_source_index"])
        ]
        extra_count = min(
            len(remaining),
            max_examples - len(sampled),
        )
        if extra_count > 0:
            extra = remaining.sample(
                n=extra_count,
                random_state=42,
            )
            sampled = pd.concat([sampled, extra])

    # Shuffle deterministically
    sampled = sampled.sample(
        n=min(len(sampled), max_examples),
        random_state=42,
    )

    examples: list[BenchmarkExample] = []

    for _, row in sampled.iterrows():
        gold = (
            GoldLabel.PASS
            if int(row["label"]) == 1
            else GoldLabel.FAIL
        )

        contamination_identifier = row.get("contamination_identifier")
        if pd.isna(contamination_identifier):
            contamination_identifier = None

        examples.append(
            BenchmarkExample(
                id=f"aggrefact_{int(row['_source_index'])}",
                task_type=TaskType.GROUNDEDNESS,
                evaluation_format=EvaluationFormat.POINTWISE,
                input=(
                    "Determine whether the following claim "
                    "is supported by the supplied document."
                ),
                output_a=row["claim"],
                output_b=None,
                context=row["doc"],
                gold_label=gold,
                source="llm_aggrefact",
                subset=row["dataset"],
                metadata={
                    "source_index": int(row["_source_index"]),
                    "contamination_identifier": contamination_identifier,
                },
            )
        )

    return examples

# ─────────────────────────────────────────
# LLMBar
# ─────────────────────────────────────────


LLMBAR_URLS = {
    "Natural": (
        "https://raw.githubusercontent.com/"
        "princeton-nlp/LLMBar/main/"
        "Dataset/LLMBar/Natural/dataset.json"
    ),
    "Adversarial_Neighbor": (
        "https://raw.githubusercontent.com/"
        "princeton-nlp/LLMBar/main/"
        "Dataset/LLMBar/Adversarial/Neighbor/dataset.json"
    ),
    "Adversarial_GPTInst": (
        "https://raw.githubusercontent.com/"
        "princeton-nlp/LLMBar/main/"
        "Dataset/LLMBar/Adversarial/GPTInst/dataset.json"
    ),
    "Adversarial_GPTOut": (
        "https://raw.githubusercontent.com/"
        "princeton-nlp/LLMBar/main/"
        "Dataset/LLMBar/Adversarial/GPTOut/dataset.json"
    ),
    "Adversarial_Manual": (
        "https://raw.githubusercontent.com/"
        "princeton-nlp/LLMBar/main/"
        "Dataset/LLMBar/Adversarial/Manual/dataset.json"
    ),
}


def load_llmbar(
    max_examples: int = 419,
) -> list[BenchmarkExample]:
    """
    LLMBar instruction-following benchmark.

    Raw fields:
        input
        output_1
        output_2
        label

    label:
        1 = output_1 is objectively better
        2 = output_2 is objectively better

    Evaluation type:
        PAIRWISE

    IMPORTANT:
        We do NOT convert output_1 into PASS/FAIL.

        LLMBar is fundamentally a pairwise judge
        benchmark, so both answers must be preserved.
    """

    # Load canonical JSON files directly.
    #
    # This avoids LLMBar's legacy Hugging Face
    # dataset-loading Python script, which modern
    # versions of `datasets` no longer execute.
    dataset = load_dataset(
        "json",
        data_files=LLMBAR_URLS,
    )

    examples: list[BenchmarkExample] = []

    # Preserve deterministic subset order.
    subset_order = [
        "Natural",
        "Adversarial_Neighbor",
        "Adversarial_GPTInst",
        "Adversarial_GPTOut",
        "Adversarial_Manual",
    ]

    for subset_name in subset_order:
        split = dataset[subset_name]

        for i, row in enumerate(split):
            if len(examples) >= max_examples:
                return examples

            label = int(row["label"])

            if label == 1:
                gold = GoldLabel.A
            elif label == 2:
                gold = GoldLabel.B
            else:
                raise ValueError(
                    f"Unexpected LLMBar label: {label}"
                )

            examples.append(
                BenchmarkExample(
                    id=f"llmbar_{subset_name}_{i}",
                    task_type=TaskType.INSTRUCTION,
                    evaluation_format=EvaluationFormat.PAIRWISE,
                    input=row["input"],
                    output_a=row["output_1"],
                    output_b=row["output_2"],
                    context=None,
                    gold_label=gold,
                    source="llmbar",
                    subset=subset_name,
                )
            )

    return examples


# ─────────────────────────────────────────
# JudgeBench
# ─────────────────────────────────────────


def _load_judgebench_split(
    split_name: str,
) -> list[BenchmarkExample]:
    """
    Load one JudgeBench split.

    JudgeBench currently provides:

        gpt
        claude

    Raw fields:
        pair_id
        original_id
        source
        question
        response_model
        response_A
        response_B
        label

    label examples:
        A>B
        B>A
    """

    dataset = load_dataset(
        "ScalerLab/JudgeBench",
        split=split_name,
    )

    examples: list[BenchmarkExample] = []

    for row in dataset:
        raw_label = row["label"].strip()

        if raw_label == "A>B":
            gold = GoldLabel.A
        elif raw_label == "B>A":
            gold = GoldLabel.B
        else:
            raise ValueError(
                f"Unexpected JudgeBench label: {raw_label}"
            )

        examples.append(
            BenchmarkExample(
                id=f"judgebench_{split_name}_{row['pair_id']}",
                task_type=TaskType.REASONING,
                evaluation_format=EvaluationFormat.PAIRWISE,
                input=row["question"],
                output_a=row["response_A"],
                output_b=row["response_B"],
                context=None,
                gold_label=gold,
                source="judgebench",
                subset=split_name,
                metadata={
                    "pair_id": row["pair_id"],
                    "original_id": row["original_id"],
                    "question_source": row["source"],
                    "response_model": row["response_model"],
                },
            )
        )

    return examples


def load_judgebench(
    max_examples: int = 350,
) -> list[BenchmarkExample]:
    """
    JudgeBench reasoning/correctness benchmark.

    JudgeBench is PAIRWISE.

    It contains two splits:

        gpt:
            350 response pairs generated by GPT-4o

        claude:
            270 response pairs generated by
            Claude 3.5 Sonnet

    We sample from BOTH splits so our benchmark does
    not accidentally contain only GPT-generated
    response pairs.
    """

    gpt_examples = _load_judgebench_split("gpt")
    claude_examples = _load_judgebench_split("claude")

    examples: list[BenchmarkExample] = []

    # Interleave the two splits:
    #
    # gpt[0]
    # claude[0]
    # gpt[1]
    # claude[1]
    # ...
    #
    # This keeps the requested subset reasonably
    # balanced when max_examples < full dataset size.
    max_len = max(
        len(gpt_examples),
        len(claude_examples),
    )

    for i in range(max_len):
        if i < len(gpt_examples):
            examples.append(gpt_examples[i])

            if len(examples) >= max_examples:
                break

        if i < len(claude_examples):
            examples.append(claude_examples[i])

            if len(examples) >= max_examples:
                break

    return examples


# ─────────────────────────────────────────
# Combined loader
# ─────────────────────────────────────────


def load_all_benchmarks(
    aggrefact_max: int = 500,
    llmbar_max: int = 419,
    judgebench_max: int = 350,
) -> list[BenchmarkExample]:
    """
    Load all benchmark datasets and normalize them
    into one list of BenchmarkExample objects.
    """

    print(
        "Loading LLM-AggreFact "
        "(groundedness / pointwise)..."
    )

    aggrefact = load_llm_aggrefact(
        aggrefact_max
    )

    print(
        f"  Loaded {len(aggrefact)} examples"
    )

    print(
        "Loading LLMBar "
        "(instruction following / pairwise)..."
    )

    llmbar = load_llmbar(
        llmbar_max
    )

    print(
        f"  Loaded {len(llmbar)} examples"
    )

    print(
        "Loading JudgeBench "
        "(reasoning / pairwise)..."
    )

    judgebench = load_judgebench(
        judgebench_max
    )

    print(
        f"  Loaded {len(judgebench)} examples"
    )

    all_examples = (
        aggrefact
        + llmbar
        + judgebench
    )

    pointwise_count = sum(
        1
        for example in all_examples
        if example.evaluation_format
        == EvaluationFormat.POINTWISE
    )

    pairwise_count = sum(
        1
        for example in all_examples
        if example.evaluation_format
        == EvaluationFormat.PAIRWISE
    )

    print()
    print(
        f"Total examples: {len(all_examples)}"
    )
    print(
        f"  Pointwise: {pointwise_count}"
    )
    print(
        f"  Pairwise:  {pairwise_count}"
    )

    return all_examples