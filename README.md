# LLM Output Arbitration System

A production evaluation layer for LLM outputs. Routes any generated response through three independent critic models in parallel, detects cross-model disagreements, and synthesises findings into a single evidence-backed verdict with confirmed issues, dismissed flags, and actionable recommendations.

Built to demonstrate the evaluation mindset required for production AI engineering — not just building systems that generate answers, but building systems that catch bad ones.

---

## The Problem

Single-model self-evaluation is structurally flawed.

When you ask the same model that generated an answer to evaluate that answer, you get a model with identical training data, identical reasoning patterns, and identical blind spots reviewing its own work. It cannot see what it cannot see.

In production AI systems this creates a dangerous failure mode: outputs that look confident and coherent but contain factual errors, broken reasoning, or critical omissions — all passing a self-evaluation check because the evaluator shares the same weaknesses as the generator.

The solution is diversity of perspective. Three models trained by different organisations on different data with different architectures will not share the same blind spots. Their disagreements surface the cases that matter most.

---

## How It Works

Input comes in as three components:
- The LLM output to evaluate
- The original prompt that generated it
- The source context (retrieved chunks if RAG)

Three critics run in parallel, each examining a different failure dimension:
- GPT-4o examines factual accuracy — are claims verifiable and internally consistent?
- Claude Haiku examines logical consistency — does the reasoning hold and do conclusions follow?
- Gemini Flash examines completeness — did the response address the full question?

Each critic returns a structured critique containing a score from 1 to 5, specific issues found with exact quotes as evidence, severity per issue, and confidence in their own assessment.

A disagreement detector then compares all three critiques, flags where critics agree (high signal), flags where they disagree (investigate further), and identifies issues one critic caught that others missed.

The adjudicator reads all three critiques and the disagreement report, weighs evidence not just opinions, resolves conflicts with explicit reasoning, and produces the final verdict.

Output is a structured verdict containing:
- Overall quality score (1-10)
- Confirmed issues with full evidence chain
- Dismissed flags with adjudicator reasoning
- Confidence level
- Actionable improvement summary

---

## Where This Sits in Production

This is not a per-query evaluation system. Running three LLM API calls plus an adjudicator on every live query is cost-prohibitive at scale. The correct positioning is as the deep investigation layer — the last resort that runs when cheaper signals have already identified a problem.

**Every live query** — free checks under 1ms:
- Latency and token count logged
- Format and schema validation
- Retrieval similarity score checked
- Basic heuristics for empty or too-short responses

**Triggered async** — runs in background, user unaffected:
- Single LLM judge when similarity score drops below 0.5
- Single LLM judge on user thumbs down

**Three-critic arbitration (this system)**:
- Runs when LLM judge score drops below 2.5/5
- Produces deep evidence chain for human review
- Handles approximately 0.5-1% of production traffic

**Scheduled batch**:
- Monthly 1% random sample audit
- Pre-deployment golden dataset validation

---

## What Makes This Different

Most evaluation tools return a number. This system returns evidence.

A weak eval returns: "Quality score: 3.2/5" — you know something is wrong but not what or why.

This system returns:

- GPT-4o flagged: claim that inflation peaked in Q2 2023 is not supported by the provided context
- Claude flagged: conclusion does not follow from the premises in paragraph 2
- Gemini flagged: question asked for three recommendations but response provided two
- Adjudicator confirmed all three issues with confidence 0.91

You know exactly what failed, which dimension, which model caught it, and which step to fix.

Evidence chains are actionable. Scores are not.

---

## What It Does Not Do

- Does not run on every live query
- Does not replace cheap programmatic checks
- Does not auto-fix issues
- Does not replace human review of verdicts
- Does not guarantee correctness when all three models share a blind spot

---

## Tech Stack

| Layer | Tool | Reason |
|-------|------|--------|
| Agent orchestration | LangGraph | Stateful parallel graph execution |
| Factual accuracy critic | GPT-4o | Strong factual grounding |
| Logical consistency critic | Claude Haiku | Strong reasoning, fast, cost-efficient |
| Completeness critic | Gemini Flash | Strong coverage, generous free tier |
| Structured outputs | Pydantic | Validated critique schemas |
| API layer | FastAPI | Production-grade async serving |
| Storage | PostgreSQL | Audit trail for every arbitration |
| Containerisation | Docker | Reproducible deployment |

---

## Architecture

```
Input: LLM output + original prompt + source context
                        |
               Parse and validate
                        |
          Parallel critic dispatch
          /             |             \
    [GPT-4o]       [Claude]       [Gemini]
    Accuracy        Logic        Completeness
          \             |             /
           Collect and compare critiques
                        |
           Disagreement detector
                        |
                  Adjudicator
                        |
            Structured verdict output
                        |
            Store to PostgreSQL
                        |
              Return to caller
```

---

## Setup

```bash
git clone https://github.com/rosmitg/llm-arbitration.git
cd llm-arbitration
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
python -m app.main
```

---

## Known Limitations

- Adds 3 to 5 seconds latency — designed for async not real-time
- Adjudicator can be wrong when all three critics share a blind spot
- Requires three API keys across three providers
- Evidence quality depends on critic prompt quality
- Not a replacement for human domain expertise on high-stakes outputs

---

## Project Status

Under active development.