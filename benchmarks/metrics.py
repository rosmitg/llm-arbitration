# benchmarks/metrics.py

import pandas as pd
from pathlib import Path
from benchmarks.runner import RESULTS_FILE


def load_df() -> pd.DataFrame:
    """Load results CSV into a DataFrame."""
    df = pd.read_csv(RESULTS_FILE)
    df = df.where(pd.notna(df), None)
    return df


def get_shared_examples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only examples where ALL three judges have results.
    Used for fair judge-vs-judge comparisons.
    Without this, GPT/Claude are evaluated on 538 examples
    while Gemini is on 499 — distorts comparisons.
    """
    counts = df.groupby("example_id")["judge"].nunique()
    shared_ids = counts[counts == 3].index
    return df[df["example_id"].isin(shared_ids)]


# ─────────────────────────────────────────
# 1. Overall accuracy per judge
# ─────────────────────────────────────────

def accuracy_per_judge(df: pd.DataFrame) -> pd.DataFrame:
    """
    How accurate is each judge overall?
    Uses all available results per judge.
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
            "correct": int(correct),
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
                "correct": int(correct),
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
# 4. Complementarity analysis
# P(B correct | A wrong) — the key research metric
# ─────────────────────────────────────────

def complementarity_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    When judge A is wrong, how often does judge B get it right?

    This is different from error overlap.
    A judge with lower standalone accuracy can still be
    the most valuable second opinion if it catches what
    the primary judge misses.

    P(B correct | A wrong) is the key Arbiter metric.
    """
    pivot = df.pivot_table(
        index="example_id",
        columns="judge",
        values="correct",
        aggfunc="first"
    ).reset_index()

    judges = [c for c in pivot.columns if c != "example_id"]
    results = []

    for j_primary in judges:
        for j_secondary in judges:
            if j_primary == j_secondary:
                continue

            # Cases where primary was wrong
            primary_wrong = pivot[pivot[j_primary] == False]

            if len(primary_wrong) == 0:
                continue

            # How often was secondary correct on those cases?
            secondary_correct = primary_wrong[j_secondary].sum()
            total = len(primary_wrong)

            results.append({
                "primary_wrong": j_primary,
                "secondary": j_secondary,
                "primary_errors": total,
                "secondary_catches": int(secondary_correct),
                "P(secondary_correct|primary_wrong)": round(
                    secondary_correct / total, 4
                )
            })

    return pd.DataFrame(results).sort_values(
        "P(secondary_correct|primary_wrong)", ascending=False
    )


# ─────────────────────────────────────────
# 5. Cheap pair routing simulation
# ─────────────────────────────────────────

def cheap_pair_analysis(df: pd.DataFrame) -> dict:
    """
    Core Arbiter hypothesis:
    If Haiku + Gemini agree → accept their verdict (no GPT call)
    If they disagree → escalate to GPT-4o

    Measures:
    - Agreement rate between cheap judges
    - Accuracy when they agree
    - GPT accuracy on disagreements
    - Simulated Arbiter accuracy vs cost
    - vs GPT-4o alone accuracy vs cost
    """
    pivot = df.pivot_table(
        index="example_id",
        columns="judge",
        values=["prediction", "correct", "estimated_cost_usd"],
        aggfunc="first"
    )
    pivot.columns = ["_".join(col) for col in pivot.columns]
    pivot = pivot.reset_index()

    gold = df[["example_id", "gold_label", "task_type"]].drop_duplicates()
    pivot = pivot.merge(gold, on="example_id")

    required = [
        "prediction_anthropic",
        "prediction_google",
        "correct_anthropic",
        "correct_google",
        "correct_openai",
        "estimated_cost_usd_anthropic",
        "estimated_cost_usd_google",
        "estimated_cost_usd_openai",
    ]

    for col in required:
        if col not in pivot.columns:
            return {"error": f"Missing column: {col} — run full benchmark first"}

    agree = pivot[
        pivot["prediction_anthropic"] == pivot["prediction_google"]
    ]
    disagree = pivot[
        pivot["prediction_anthropic"] != pivot["prediction_google"]
    ]

    agree_total = len(agree)
    total_examples = len(pivot)

    agree_correct = int(agree["correct_anthropic"].sum())
    agree_accuracy = agree_correct / agree_total if agree_total > 0 else 0

    gpt_on_disagree = (
        disagree["correct_openai"].mean()
        if len(disagree) > 0 else 0
    )

    # Simulate Arbiter routing
    arbiter_correct = 0
    arbiter_cost = 0.0

    for _, row in pivot.iterrows():
        haiku_pred = row["prediction_anthropic"]
        gemini_pred = row["prediction_google"]
        haiku_correct = row["correct_anthropic"]
        gpt_correct = row["correct_openai"]

        # Always pay for both cheap judges
        arbiter_cost += (
            row["estimated_cost_usd_anthropic"] +
            row["estimated_cost_usd_google"]
        )

        if haiku_pred == gemini_pred:
            # Agree → use cheap verdict
            if haiku_correct:
                arbiter_correct += 1
        else:
            # Disagree → escalate to GPT-4o
            arbiter_cost += row["estimated_cost_usd_openai"]
            if gpt_correct:
                arbiter_correct += 1

    arbiter_accuracy = arbiter_correct / total_examples if total_examples > 0 else 0

    gpt_alone_accuracy = df[df["judge"] == "openai"]["correct"].mean()
    gpt_alone_cost = df[df["judge"] == "openai"]["estimated_cost_usd"].sum()

    escalation_rate = len(disagree) / total_examples if total_examples > 0 else 0

    return {
        "total_examples":                    total_examples,
        "agree_cases":                       agree_total,
        "disagree_cases":                    len(disagree),
        "agreement_rate":                    round(agree_total / total_examples, 4),
        "escalation_rate":                   round(escalation_rate, 4),
        "cheap_agree_accuracy":              round(agree_accuracy, 4),
        "gpt_accuracy_on_disagreements":     round(float(gpt_on_disagree), 4),
        "arbiter_simulated_accuracy":        round(arbiter_accuracy, 4),
        "gpt_alone_accuracy":                round(float(gpt_alone_accuracy), 4),
        "arbiter_cost_usd":                  round(arbiter_cost, 4),
        "gpt_alone_cost_usd":                round(gpt_alone_cost, 4),
        "cost_saving_pct":                   round(
            1 - arbiter_cost / gpt_alone_cost, 4
        ) if gpt_alone_cost > 0 else 0,
    }


# ─────────────────────────────────────────
# 6. Full report
# ─────────────────────────────────────────

def print_report():
    df = load_df()
    shared = get_shared_examples(df)

    print(f"\n{'='*60}")
    print("ARBITER CALIBRATION REPORT")
    print(f"{'='*60}")
    print(f"Total results:    {len(df)}")
    print(f"Unique examples:  {df['example_id'].nunique()}")
    print(f"Shared examples:  {shared['example_id'].nunique()} (all 3 judges)")

    print(f"\n{'─'*60}")
    print("1. STANDALONE ACCURACY (all available per judge)")
    print(f"{'─'*60}")
    print(accuracy_per_judge(df).to_string(index=False))

    print(f"\n{'─'*60}")
    print("2. ACCURACY BY TASK TYPE (shared examples)")
    print(f"{'─'*60}")
    task_acc = accuracy_per_judge_per_task(shared)
    pivot_task = task_acc.pivot_table(
        index="judge",
        columns="task_type",
        values="accuracy"
    )
    print(pivot_task.to_string())

    print(f"\n{'─'*60}")
    print("3. ERROR OVERLAP (shared examples)")
    print(f"{'─'*60}")
    overlap = error_overlap(shared)
    print(overlap[[
        "judge_1", "judge_2", "errors_j1", "errors_j2",
        "both_wrong", "either_wrong", "overlap_pct"
    ]].to_string(index=False))

    print(f"\n{'─'*60}")
    print("4. COMPLEMENTARITY — P(B correct | A wrong)")
    print(f"{'─'*60}")
    comp = complementarity_analysis(shared)
    print(comp.to_string(index=False))

    print(f"\n{'─'*60}")
    print("5. CHEAP PAIR ROUTING SIMULATION (shared examples)")
    print(f"{'─'*60}")
    routing = cheap_pair_analysis(shared)
    for k, v in routing.items():
        print(f"  {k:40} {v}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    print_report()