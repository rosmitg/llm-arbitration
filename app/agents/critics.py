# app/agents/critics.py
# Critic agents for the LLM Output Arbitration System.
# One function handles all three providers — same pattern, different model.
# Each critic evaluates a different dimension independently.

import os
import json
import anthropic
import openai
from google import genai as google_genai
from dotenv import load_dotenv
from app.models import (
    ModelVersion,
    Dimension,
    Critique,
    Issue,
    Severity,
    ArbitrationInput
)

load_dotenv()


def call_model(model_version: ModelVersion, prompt: str) -> str:
    """
    Routes prompt to the correct LLM provider.
    Returns raw text response.
    Abstracts away provider differences so run_critic
    doesn't care which model it's talking to.
    """
    if model_version == ModelVersion.GPT4O:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    elif model_version == ModelVersion.CLAUDE_HAIKU:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    elif model_version == ModelVersion.GEMINI_FLASH:
        client = google_genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        chat = client.chats.create(model="gemini-3.5-flash-lite")
        response = chat.send_message(prompt)
        return response.text

    else:
        raise ValueError(f"Unknown model version: {model_version}")


def build_critic_prompt(
    input: ArbitrationInput,
    dimension: Dimension
) -> str:
    """
    Builds the evaluation prompt for a critic.
    Same template every time — only the instruction changes per dimension.
    Returns ONLY JSON — no markdown, no explanation.
    """
    dimension_instructions = {
        Dimension.ACCURACY: (
            "You are evaluating FACTUAL ACCURACY. "
            "Check whether all claims in the output are verifiable "
            "and consistent with the source context provided. "
            "Flag any claim that cannot be supported by the context."
        ),
        Dimension.LOGIC: (
            "You are evaluating LOGICAL CONSISTENCY. "
            "Check whether the reasoning holds, conclusions follow "
            "from the premises, and there are no contradictions. "
            "Flag any logical gaps or unsupported conclusions."
        ),
        Dimension.COMPLETENESS: (
            "You are evaluating COMPLETENESS. "
            "Check whether the output fully addresses the original prompt. "
            "Flag any parts of the question that were ignored or only "
            "partially answered."
        )
    }

    return f"""
You are an expert evaluator. {dimension_instructions[dimension]}

ORIGINAL PROMPT:
{input.original_prompt}

SOURCE CONTEXT:
{input.source_context or "No source context provided."}

LLM OUTPUT TO EVALUATE:
{input.llm_output}

Evaluate the output strictly on your assigned dimension.
Be specific — quote exact text when flagging issues.

Return ONLY valid JSON, no markdown, no explanation:
{{
    "passed": true or false,
    "faithfulness_score": 0.0 to 1.0,
    "relevance_score": 0.0 to 1.0,
    "confidence_score": 0.0 to 1.0,
    "issues": [
        {{
            "description": "what is wrong",
            "quote": "exact quote from the output",
            "severity": "low | medium | critical",
            "dimension": "{dimension.value}"
        }}
    ]
}}

If no issues found return an empty issues array.
"""


def run_critic(
    input: ArbitrationInput,
    model_version: ModelVersion,
    dimension: Dimension
) -> Critique:
    """
    Runs a single critic evaluation.
    Builds prompt → calls model → parses response → returns Critique.
    """
    prompt = build_critic_prompt(input, dimension)
    raw = call_model(model_version, prompt)

    # Strip markdown fences if model adds them
    content = raw.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    data = json.loads(content)

    issues = [
        Issue(
            description=issue["description"],
            quote=issue["quote"],
            severity=Severity(issue["severity"]),
            dimension=Dimension(issue["dimension"])
        )
        for issue in data.get("issues", [])
    ]

    return Critique(
        model_version=model_version,
        dimension=dimension,
        passed=data["passed"],
        issues=issues,
        faithfulness_score=data["faithfulness_score"],
        relevance_score=data["relevance_score"],
        confidence_score=data["confidence_score"]
    )