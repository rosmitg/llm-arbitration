# benchmarks/metrics.py

import pandas as pd
from pathlib import Path
from benchmarks.runner import RESULTS_FILE


def load_df() -> pd.DataFrame:
    """Load results CSV into a DataFrame."""
    df = pd.read_csv(RESULTS_FILE)
    df = df.where(pd.notna(df), None)
    return df


# ─────────────────────────────────────────
# 1. Overall accuracy per judge
# ─────────────────────────────────────────

def accuracy_per_judge(df: pd.DataFrame) -> pd.DataFrame:
    """
    How accurate is each judge overall?
    """
    results = []
    for judge in df["judge"].unique():
        subset = df[df["judge"] == judge]
        correct = subset["correct"].sum()
        total = len(subset)
        cost = subset["estimated_cost_usd"].sum()
        avg_latency = subset["latency_ms"].mean()
        results.append({
            "judge": judge,
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total, 4),
            "total_cost_usd": round(cost, 4),
            "avg_latency_ms": round(avg_latency, 1)
        })
    return pd.DataFrame(results).sort_values("accuracy", ascending=False)


# ─────────────────────────────────────────
# 2. Accuracy per judge per task type
# ─────────────────────────────────────────

def accuracy_per_judge_per_task(df: pd.DataFrame) -> pd.DataFrame:
    """
    Where does each judge specialise?
    Groundedness vs instruction vs reasoning.
    """
    results = []
    for judge in df["judge"].unique():
        for task in df["task_type"].unique():
            subset = df[
                (df["judge"] == judge) &
                (df["task_type"] == task)
            ]
            if len(subset) == 0:
                continue
            correct = subset["correct"].sum()
            total = len(subset)
            results.append({
                "judge": judge,
                "task_type": task,
                "correct": correct,
                "total": total,
                "accuracy": round(correct / total, 4),
            })
    return pd.DataFrame(results).sort_values(
        ["task_type", "accuracy"], ascending=[True, False]
    )


# ─────────────────────────────────────────
# 3. Error overlap between judges
# ─────────────────────────────────────────

def error_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """
    When judge A is wrong, how often is judge B also wrong?
    Low overlap = complementary judges.
    High overlap = correlated errors, jury adds little value.
    """
    # Get error sets per judge
    errors = {}
    for judge in df["judge"].unique():
        subset = df[df["judge"] == judge]
        errors[judge] = set(
            subset[subset["correct"] == False]["example_id"]
        )

    judges = list(errors.keys())
    results = []

    for i, j1 in enumerate(judges):
        for j2 in judges[i+1:]:
            both_wrong = errors[j1] & errors[j2]
            either_wrong = errors[j1] | errors[j2]
            only_j1 = errors[j1] - errors[j2]
            only_j2 = errors[j2] - errors[j1]

            overlap = (
                len(both_wrong) / len(either_wrong)
                if either_wrong else 0
            )

            results.append({
                "judge_1": j1,
                "judge_2": j2,
                "errors_j1": len(errors[j1]),
                "errors_j2": len(errors[j2]),
                "both_wrong": len(both_wrong),
                "either_wrong": len(either_wrong),
                "only_j1_wrong": len(only_j1),
                "only_j2_wrong": len(only_j2),
                "overlap_pct": round(overlap, 4),
            })

    return pd.DataFrame(results).sort_values("overlap_pct")


# ─────────────────────────────────────────
# 4. Cheap pair agreement analysis
# ─────────────────────────────────────────

def cheap_pair_analysis(df: pd.DataFrame) -> dict:
    """
    When Haiku and Gemini agree, how often are they right?
    When they disagree, how often does GPT-4o resolve correctly?

    This is the core Arbiter routing hypothesis.
    """
    # Pivot so each example has one row per judge
    pivot = df.pivot_table(
        index="example_id",
        columns="judge",
        values=["prediction", "correct", "estimated_cost_usd"],
        aggfunc="first"
    )
    pivot.columns = ["_".join(col) for col in pivot.columns]
    pivot = pivot.reset_index()

    # Merge gold label
    gold = df[["example_id", "gold_label", "task_type"]].drop_duplicates()
    pivot = pivot.merge(gold, on="example_id")

    # Cases where both cheap judges made a prediction
    has_cheap = (
        pivot.get("prediction_anthropic") is not None and
        pivot.get("prediction_google") is not None
    )

    cheap_cols = [c for c in pivot.columns if "prediction_anthropic" in c or "prediction_google" in c]

    if "prediction_anthropic" not in pivot.columns or "prediction_google" not in pivot.columns:
        return {"error": "Missing cheap judge columns — run full benchmark first"}

    # Agreement cases
    agree = pivot[
        pivot["prediction_anthropic"] == pivot["prediction_google"]
    ]
    disagree = pivot[
        pivot["prediction_anthropic"] != pivot["prediction_google"]
    ]

    # Accuracy when cheap pair agrees
    agree_correct = agree["correct_anthropic"].sum() if "correct_anthropic" in agree.columns else 0
    agree_total = len(agree)
    agree_accuracy = agree_correct / agree_total if agree_total > 0 else 0

    # GPT-4o accuracy on disagreement cases
    gpt_on_disagree = 0
    if "correct_openai" in disagree.columns:
        gpt_on_disagree = disagree["correct_openai"].mean()

    # Simulated Arbiter routing:
    # If cheap pair agrees → use their verdict
    # If cheap pair disagrees → use GPT-4o
    arbiter_correct = 0
    arbiter_cost = 0.0
    gpt_cost_per_call = df[df["judge"] == "openai"]["estimated_cost_usd"].mean()
    cheap_cost_per_call = df[df["judge"] != "openai"]["estimated_cost_usd"].mean()

    for _, row in pivot.iterrows():
        if "prediction_anthropic" not in pivot.columns:
            break
        haiku_pred = row.get("prediction_anthropic")
        gemini_pred = row.get("prediction_google")
        gpt_correct = row.get("correct_openai")
        haiku_correct = row.get("correct_anthropic")

        # Cost: always pay for both cheap judges
        arbiter_cost += cheap_cost_per_call * 2

        if haiku_pred == gemini_pred:
            # Agree — use cheap verdict
            if haiku_correct:
                arbiter_correct += 1
        else:
            # Disagree — escalate to GPT-4o
            arbiter_cost += gpt_cost_per_call
            if gpt_correct:
                arbiter_correct += 1

    total_examples = len(pivot)
    arbiter_accuracy = arbiter_correct / total_examples if total_examples > 0 else 0

    # GPT-4o alone accuracy and cost
    gpt_alone_accuracy = df[df["judge"] == "openai"]["correct"].mean()
    gpt_alone_cost = df[df["judge"] == "openai"]["estimated_cost_usd"].sum()

    return {
        "total_examples": total_examples,
        "agree_cases": agree_total,
        "disagree_cases": len(disagree),
        "agree_pct": round(agree_total / total_examples, 4) if total_examples > 0 else 0,
        "cheap_agree_accuracy": round(agree_accuracy, 4),
        "gpt_accuracy_on_disagreements": round(gpt_on_disagree, 4),
        "arbiter_accuracy": round(arbiter_accuracy, 4),
        "arbiter_cost_usd": round(arbiter_cost, 4),
        "gpt_alone_accuracy": round(gpt_alone_accuracy, 4),
        "gpt_alone_cost_usd": round(gpt_alone_cost, 4),
        "cost_saving_pct": round(
            1 - arbiter_cost / gpt_alone_cost, 4
        ) if gpt_alone_cost > 0 else 0,
    }


# ─────────────────────────────────────────
# 5. Full report
# ─────────────────────────────────────────

def print_report():
    """Print the full calibration report."""
    df = load_df()
    total = len(df)
    examples = df["example_id"].nunique()

    print(f"\n{'='*60}")
    print("ARBITER CALIBRATION REPORT")
    print(f"{'='*60}")
    print(f"Total results: {total}")
    print(f"Unique examples: {examples}")
    print(f"Judges: {df['judge'].nunique()}")

    print(f"\n{'─'*60}")
    print("1. OVERALL ACCURACY")
    print(f"{'─'*60}")
    acc = accuracy_per_judge(df)
    print(acc.to_string(index=False))

    print(f"\n{'─'*60}")
    print("2. ACCURACY BY TASK TYPE")
    print(f"{'─'*60}")
    task_acc = accuracy_per_judge_per_task(df)
    pivot = task_acc.pivot_table(
        index="judge",
        columns="task_type",
        values="accuracy"
    )
    print(pivot.to_string())

    print(f"\n{'─'*60}")
    print("3. ERROR OVERLAP")
    print(f"{'─'*60}")
    overlap = error_overlap(df)
    print(overlap[["judge_1", "judge_2", "both_wrong",
                   "either_wrong", "overlap_pct"]].to_string(index=False))

    print(f"\n{'─'*60}")
    print("4. CHEAP PAIR ROUTING SIMULATION")
    print(f"{'─'*60}")
    routing = cheap_pair_analysis(df)
    for k, v in routing.items():
        print(f"  {k:35} {v}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    print_report()