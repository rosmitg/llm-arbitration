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