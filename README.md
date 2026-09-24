# LLM Arbiter

**LLM Arbiter is an experimental evaluation framework for studying when different evaluators add useful information — and when they do not.**

The project began with a simple hypothesis:

> If several LLM judges evaluate the same output, their different failure modes may make the combined system more reliable than relying on a single judge.

Initial experiments showed that this idea is only partly true.

Different judges do make different mistakes, but their errors are also substantially correlated. Simple majority voting and disagreement-based escalation do not automatically outperform the best individual evaluator.

That result changed the direction of the project.

The current research question is broader:

> **Can an evaluation system choose the right evaluator for the type of problem, using general-purpose LLM judges, specialised verifiers, deterministic tools, and selective escalation only when they add useful information?**

---

## Why this project exists

LLM evaluation is often implemented using one of two approaches:

1. use one powerful LLM as a judge;
2. use several LLM judges and aggregate their answers.

Both approaches have weaknesses.

A single judge can make systematic mistakes.

Multiple judges can also fail together because different LLM families may share similar biases and reasoning failures.

Running several expensive models on every output also increases cost and latency without guaranteeing better evaluation quality.

LLM Arbiter explores a different approach:

```text
evaluation task
      ↓
identify what needs to be checked
      ↓
choose the most appropriate evaluator
      ↓
evaluate
      ↓
escalate only when additional evaluation
is expected to provide useful information
```

The evaluator does not necessarily need to be another LLM.

Examples could eventually include:

```text
Groundedness
→ factual consistency / entailment verifier

Numerical accuracy
→ calculator / Python / SQL

Code correctness
→ compiler / tests / sandbox

Structured output
→ JSON/schema validator

Instruction following
→ LLM judge / constraint checker

Semantic reasoning
→ general-purpose LLM judge

High-risk unresolved case
→ stronger LLM or human review
```

---

# Current Experiment

The current benchmark compares three LLM judges:

```text
Anthropic  → Claude Haiku
Google     → Gemini Flash
OpenAI     → GPT-4o
```

These models intentionally represent different cost/capability points.

The experiment is therefore **not intended to rank OpenAI, Anthropic, and Google as providers**.

Instead, it studies:

* standalone evaluator accuracy;
* performance across different task types;
* disagreement between judges;
* overlapping errors;
* whether one judge can recover another judge's mistakes;
* whether cheap evaluators can reduce expensive-model usage;
* whether simple arbitration actually improves evaluation quality.

---

# Benchmarks

Three benchmark families are currently normalised into a common `BenchmarkExample` representation.

## LLM-AggreFact

Used for **groundedness / factual-support evaluation**.

The evaluator receives:

```text
document
+
claim
```

and determines whether the claim is supported by the supplied evidence.

The initial loader accidentally selected the first 500 rows of the dataset, which were all from `AggreFact-CNN`.

This was discovered during analysis and fixed.

Groundedness examples are now sampled with a fixed seed across 11 underlying datasets:

```text
RAGTruth
ExpertQA
TofuEval-MeetB
TofuEval-MediaS
WiCE
AggreFact-CNN
FactCheck-GPT
LFQA
AggreFact-XSum
ClaimVerify
Reveal
```

The 500-example stratified sample currently contains approximately:

```text
255 supported
245 unsupported
```

This prevents the groundedness experiment from being dominated by one source dataset or one label.

---

## LLMBar

Used for **instruction-following evaluation**.

LLMBar is preserved as a pairwise benchmark:

```text
instruction
+
response A
+
response B
      ↓
which response better follows the instruction?
```

The benchmark is not flattened into artificial PASS/FAIL labels.

Its natural and adversarial subsets are retained for later slice analysis.

---

## JudgeBench

Used for **reasoning / correctness evaluation**.

JudgeBench is also treated as a pairwise benchmark:

```text
question
+
response A
+
response B
      ↓
which response is better?
```

Both available response-generation splits are retained.

---

# Benchmark Infrastructure

The project currently includes:

```text
✅ unified Pydantic benchmark schema
✅ pointwise evaluation support
✅ pairwise evaluation support

✅ LLM-AggreFact loader
✅ LLMBar loader
✅ JudgeBench loader

✅ OpenAI judge integration
✅ Anthropic judge integration
✅ Google judge integration

✅ resumable benchmark execution
✅ completed-call detection
✅ CSV result persistence
✅ latency tracking
✅ token/cost tracking

✅ standalone accuracy analysis
✅ accuracy by task type
✅ pairwise error-overlap analysis
✅ conditional error-recovery analysis
✅ cheap-pair routing simulation
```

A benchmark run can be resumed without paying for already completed evaluations.

```bash
python -m benchmarks.runner
```

Analysis can be generated with:

```bash
python -m benchmarks.metrics
```

---

# Current Results

The current result set contains:

```text
5,136 judge results
1,730 unique benchmark examples
1,676 examples evaluated by all three judges
```

For fair judge-to-judge comparisons, the analysis below uses only the **1,676 shared examples**.

---

## Standalone Performance

Approximate accuracy on the shared evaluation set:

```text
Gemini Flash      ~81.7%
GPT-4o            ~80.3%
Claude Haiku      ~79.1%
```

The important result is not that one provider "won".

Rather:

> **The strongest individual evaluator in this experiment was a relatively inexpensive model, and the more expensive judge did not automatically provide higher evaluation accuracy.**

This means model cost and judge quality cannot simply be assumed to increase together.

---

# Performance by Task

The judges also behaved differently across evaluation types.

```text
                Groundedness   Instruction   Reasoning

Claude Haiku       84.6%          77.5%        64.5%

Gemini Flash       85.9%          79.4%        72.2%

GPT-4o             83.8%          82.3%        66.8%
```

In this benchmark run:

* Gemini performed best on groundedness.
* Gemini performed best on reasoning.
* GPT-4o performed best on instruction following.
* Claude Haiku had the lowest standalone accuracy across these three aggregated task categories.

These results suggest **task-dependent evaluator performance**, but they are not yet treated as a learned routing policy.

A proper router must be calibrated on one split and evaluated on unseen held-out examples.

---

# Error Complementarity

Standalone accuracy does not tell the whole story.

A weaker judge may still be useful if it catches errors made by another judge.

For example:

```text
When Claude was wrong:
Gemini was correct ~45% of the time.

When GPT-4o was wrong:
Gemini was correct ~39% of the time.

When Gemini was wrong:
Claude was correct ~37% of the time.

When Gemini was wrong:
GPT-4o was correct ~34% of the time.
```

This shows that the judges contain some complementary information.

However, their failures are far from independent.

Pairwise error overlap remains substantial:

```text
Claude ↔ Gemini     ~42%
GPT-4o ↔ Gemini     ~46%
Claude ↔ GPT-4o     ~50%
```

Therefore:

> **Different model providers should not be assumed to provide independent votes.**

---

# Cheap-Pair Escalation Experiment

A simple routing strategy was tested:

```text
Claude Haiku
      +
Gemini Flash
      ↓
Do they agree?
   /       \
 yes        no
  ↓          ↓
accept      GPT-4o
cheap       escalation
verdict
```

Across the 1,676 shared examples:

```text
Cheap judges agreed:       83.8%
Cheap judges disagreed:    16.2%

Accuracy when they agreed: 86.3%

GPT-4o accuracy on
disagreement cases:        56.8%
```

The resulting simulated system achieved:

```text
Arbiter cascade accuracy:  ~81.5%
GPT-4o alone:              ~80.3%
```

Measured cost:

```text
Arbiter cascade:  ~$1.07
GPT-4o alone:     ~$4.71
```

The cascade therefore used GPT-4o on only about 16% of cases and reduced measured cost by approximately 77%.

However, this was **not the winning strategy**.

Gemini Flash alone achieved approximately:

```text
81.7%
```

which slightly exceeded the cascade while requiring much less system complexity.

This is an important negative result:

> **Naive multi-model arbitration did not outperform the strongest individual evaluator in the experiment.**

---

# False Consensus

Disagreement is not the only failure mode.

A particularly important class of errors occurs when several judges agree with each other but are still wrong.

During the initial AggreFact-CNN experiment, 37 cheap-pair false-consensus cases were identified.

In those examples:

```text
Claude wrong
+
Gemini wrong
+
both produced the same verdict
```

More importantly:

```text
GPT-4o was also wrong on 31 of the 37 cases.
```

This suggests that some errors are **correlated across several general-purpose LLM families**.

Simply adding a larger LLM may therefore fail to resolve the hardest cases.

This observation is one of the main reasons the project's direction is expanding beyond multi-LLM voting.

---

# What We Learned

The initial hypothesis was approximately:

```text
multiple diverse LLM judges
→ disagreement detection
→ stronger adjudicator
→ more reliable evaluation
```

The benchmark results suggest a more complicated reality.

### 1. More judges do not automatically produce better evaluation

A single inexpensive evaluator currently performs slightly better than the simple multi-model cascade.

### 2. Different models do provide complementary information

One model can correctly evaluate examples another model gets wrong.

### 3. Errors remain strongly correlated

Different providers cannot be treated as statistically independent judges.

### 4. Agreement is not proof of correctness

Several models can confidently converge on the same wrong answer.

### 5. Stronger models are not automatically good adjudicators

GPT-4o achieved only about 57% accuracy on the examples where the two cheaper judges disagreed.

### 6. Evaluator selection may matter more than evaluator quantity

The judges showed different relative performance across groundedness, instruction following, and reasoning.

This suggests that the important question may not be:

> "How many judges should evaluate this?"

but instead:

> **"Which evaluator is appropriate for this particular failure mode?"**

---

# New Research Direction

LLM Arbiter is therefore moving from:

> **multi-LLM voting**

toward:

> **task-aware and risk-aware evaluator orchestration.**

The long-term system could select between several kinds of evaluators:

```text
                    ┌─ specialised verifier
                    │
evaluation request ─┼─ general LLM judge
                    │
                    ├─ deterministic checker
                    │
                    ├─ stronger LLM
                    │
                    └─ human review
```

The goal becomes:

> **Use the cheapest evaluator that is sufficiently reliable for the current problem, and escalate only when another evaluator is expected to provide meaningful additional information.**

---

# Open Research Questions

The next experiments will focus on the following questions.

## 1. Can task-aware routing beat the best single judge?

Current data suggests different judges perform differently across task categories.

A simple hypothesis is:

```text
Groundedness
→ best calibrated groundedness evaluator

Instruction following
→ best calibrated instruction evaluator

Reasoning
→ best calibrated reasoning evaluator
```

This must be evaluated using separate calibration and held-out test sets to avoid overfitting.

---

## 2. Can we predict false consensus?

The difficult cases are not always disagreements.

Sometimes:

```text
Judge A = PASS
Judge B = PASS
Gold    = FAIL
```

The system therefore needs to estimate:

```text
P(consensus is wrong | observable features)
```

Possible signals include:

* evaluation task type;
* dataset/domain;
* verdict type;
* document length;
* answer length;
* numerical content;
* negation;
* evidence quality;
* historical error rate for similar examples;
* structured evaluator confidence signals.

---

## 3. Can specialised evaluators catch failures shared by general LLMs?

This is one of the most important next experiments.

For example, groundedness evaluation could compare:

```text
general-purpose LLM judge

vs

specialised factual-consistency /
entailment verifier
```

If specialised evaluators catch errors that GPT, Claude, and Gemini all share, this would provide a stronger form of evaluator diversity than simply adding another general-purpose LLM.

---

## 4. Can deterministic tools outperform LLM judges on verifiable dimensions?

Some evaluation problems do not require another language model.

Examples:

```text
numerical accuracy
→ calculator

structured output
→ schema validator

code correctness
→ compiler / tests

SQL
→ execute query

citation existence
→ citation verifier
```

A future Arbiter could route these dimensions directly to deterministic evaluators.

---

## 5. What is the best quality-cost frontier?

Evaluation strategies should be compared not only by accuracy but also by:

```text
accuracy
cost
latency
calls per example
expensive-model usage
error recovery
false-consensus rate
```

The best system may not be the most accurate evaluator in isolation.

Instead, it may be the policy that delivers the highest evaluation reliability for a given cost or latency budget.

---

## 6. When is a single judge actually the correct answer?

Arbiter should not assume that arbitration is always useful.

If calibration shows:

```text
one evaluator
≈ best accuracy
+ lowest cost
+ acceptable risk
```

then the correct policy should simply be:

```text
use that evaluator
```

The system should add complexity only when the benchmark demonstrates that additional evaluation provides meaningful value.

---

# Research Principle

The project deliberately treats negative experimental results as useful information.

LLM Arbiter is not designed to prove that multiple judges are always superior.

Instead, it asks:

> **When does another evaluator actually provide useful information?**

and ultimately:

> **Can we choose the right evaluator, verifier, or escalation path for each type of LLM failure?**

---

# Current Status

```text
Benchmark infrastructure            ✅
Three-provider judge runner         ✅
Resumable API execution             ✅
Cost / latency tracking             ✅
Groundedness benchmark sampling     ✅
Instruction benchmark               ✅
Reasoning benchmark                 ✅
Error-overlap analysis              ✅
Complementarity analysis            ✅
Simple routing simulation           ✅

Held-out routing experiment         ⏳
False-consensus risk modelling      ⏳
Specialised verifier comparison     ⏳
Deterministic evaluator support     ⏳
Evaluator selection policy          ⏳
Production API / middleware         Later
```

---

# Project Direction

The first phase asked:

> Can several LLM judges outperform one?

The next phase asks a more useful question:

> **Can we determine which evaluator should be trusted for which kind of problem — and avoid paying for additional evaluation when it does not add useful information?**

That is the direction LLM Arbiter is now exploring.
