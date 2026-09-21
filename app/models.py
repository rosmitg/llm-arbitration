# app/models.py
# Core data models for the LLM Output Arbitration System.
# Every component in the pipeline communicates through these models.
# Pydantic validates all data at every boundary — no silent failures.

from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


# ─────────────────────────────────────────
# Enums — constrained value sets
# Using str, Enum so values serialise cleanly to JSON
# and LLM outputs can be validated against them directly
# ─────────────────────────────────────────

class Severity(str, Enum):
    """How serious is the issue found by a critic."""
    LOW = "low"
    MEDIUM = "medium"
    CRITICAL = "critical"


class Dimension(str, Enum):
    """Which evaluation dimension this relates to.
    Each critic is responsible for exactly one dimension."""
    ACCURACY = "accuracy"       # GPT-4o: are facts correct?
    COMPLETENESS = "completeness"  # Gemini: did it answer fully?
    LOGIC = "logic"             # Claude: does reasoning hold?


class ModelVersion(str, Enum):
    """The three critic models used in the arbitration system.
    Different providers = different training data = different blind spots.
    This diversity is the core mechanism of the system."""
    GPT4O = "gpt-4o"
    CLAUDE_HAIKU = "claude-haiku-4-5-20251001"
    GEMINI_FLASH = "gemini-2.0-flash-lite"


# ─────────────────────────────────────────
# Core models
# ─────────────────────────────────────────

class Issue(BaseModel):
    """A single specific problem found by a critic.
    Contains exact evidence — not just a vague score.
    This is what makes the system actionable vs a number."""
    description: str    # what exactly is wrong
    quote: str          # exact quote from the LLM output as evidence
    severity: Severity  # how serious is this issue
    dimension: Dimension  # which evaluation dimension this falls under


class Critique(BaseModel):
    """The structured output from a single critic model.
    Each critic evaluates one dimension independently.
    Running three critics in parallel catches what one would miss."""
    model_version: ModelVersion   # which model produced this critique
    dimension: Dimension          # which dimension this critic evaluated
    passed: bool                  # did the output pass this dimension?
    issues: list[Issue]           # specific problems found with evidence
    faithfulness_score: float = Field(ge=0.0, le=1.0)   # claims grounded in context?
    relevance_score: float = Field(ge=0.0, le=1.0)      # relevant to the prompt?
    confidence_score: float = Field(ge=0.0, le=1.0)     # how confident is this critic?


class ArbitrationInput(BaseModel):
    """Everything needed to evaluate an LLM output.
    Source context is optional — works for both RAG and non-RAG systems."""
    llm_output: str                        # the generated text to evaluate
    original_prompt: str                   # what was asked
    source_context: Optional[str] = None  # retrieved context if RAG pipeline


class Verdict(BaseModel):
    """The final adjudicator output after resolving all critic disagreements.
    Confirmed issues = critics agreed. Dismissed flags = overruled with reasoning.
    Evidence chains make this actionable — not just a score."""
    overall_quality_score: float = Field(ge=0.0, le=10.0)  # 1-10 quality rating
    issues: list[Issue]           # confirmed problems backed by evidence
    dismissed_flags: list[Issue]  # raised by a critic but overruled by adjudicator
    confidence_score: float = Field(ge=0.0, le=1.0)  # adjudicator confidence
    summary: str                  # actionable one-paragraph improvement summary


class ArbitrationState(BaseModel):
    """LangGraph state object — flows through every node in the graph.
    Starts with just the input. Each node adds to it.
    By the end it holds the complete arbitration record."""
    input: ArbitrationInput           # what we're evaluating
    critiques: list[Critique] = []    # collected as critics complete (parallel)
    disagreements: list[Issue] = []   # conflicts detected between critics
    verdict: Optional[Verdict] = None  # None until adjudicator runs

    class Config:
        # Required for LangGraph compatibility
        # Allows complex nested Pydantic types in state
        arbitrary_types_allowed = True