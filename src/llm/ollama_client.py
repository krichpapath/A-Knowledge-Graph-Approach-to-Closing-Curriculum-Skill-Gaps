# File: src/llm/ollama_client.py

import requests
import json
from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/generate"

def call_ollama(model: str, prompt: str, temperature: float = 0.2) -> str:
    """
    Call local Ollama model and return raw text response (string).
    We assume Ollama is running locally: ollama serve
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("response", "")
    return text


def extract_json_from_response(text: str) -> Optional[dict]:
    """
    Try to extract JSON object from LLM response.
    Handles code fences and extra text.
    """
    text = text.strip()

    # Remove markdown fences if present
    if text.startswith("```"):
        # remove first ```...``` block markers
        text = text.strip("`")
        # Sometimes they write ```json
        text = text.replace("json", "", 1).strip()

    # Try to locate first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    json_str = text[start:end+1]
    try:
        return json.loads(json_str)
    except Exception:
        return None
