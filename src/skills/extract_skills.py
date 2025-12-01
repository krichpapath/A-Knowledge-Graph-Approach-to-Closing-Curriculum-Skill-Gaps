# File: src/skills/extract_skills.py

from typing import List, Dict
from src.llm.ollama_client import call_ollama, extract_json_from_response

SYSTEM_INSTRUCTION = """
You are an expert course analyst.
Given a course title and description, extract a list of core SKILLS taught in this course.

Return EXACTLY this JSON format:

{
  "skills": [
    {
      "name": "short English name of the skill",
      "description": "1-2 sentence description of what this skill is about."
    }
  ]
}

Rules:
- 5 to 12 skills per course.
- Skills should be atomic concepts (e.g., "Boolean Algebra", "Set Theory", "Graph Theory").
- DO NOT include general words like "problem solving" unless it is explicitly taught.
- Only output JSON, no explanations, no commentary.
"""

USER_TEMPLATE = """
Course Title: {title}
Course Description: {description}
"""

def extract_skills_for_course(
    model: str,
    title: str,
    description: str
) -> List[Dict[str, str]]:
    """
    Use local LLM to extract skills for a single course.
    Returns a list of dicts: {"name": str, "description": str}
    """
    prompt = SYSTEM_INSTRUCTION + "\n\n" + USER_TEMPLATE.format(
        title=title, description=description
    )
    raw = call_ollama(model=model, prompt=prompt, temperature=0.1)
    data = extract_json_from_response(raw)
    if not data:
        return []

    skills = data.get("skills", [])
    cleaned = []
    for s in skills:
        name = (s.get("name") or "").strip()
        desc = (s.get("description") or "").strip()
        if not name:
            continue
        cleaned.append({"name": name, "description": desc})
    return cleaned
