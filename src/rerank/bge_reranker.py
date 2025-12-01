# File: src/rerank/bge_reranker.py

from typing import List, Tuple
from sentence_transformers import CrossEncoder
from src.config import RERANKER_MODEL_NAME

_reranker = None

def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def rerank_job_skills(
    job_text: str,
    skills: List[Tuple[str, str, float]]
) -> List[Tuple[str, str, float, float]]:
    """
    Re-rank candidate skills for a job.

    skills: list of (skill_name, skill_desc, embed_score)
    returns: list of (skill_name, skill_desc, embed_score, rerank_score)
            sorted descending by rerank_score
    """
    if not skills:
        return []

    # Build pairs: (job_text, skill_text)
    pairs = []
    for name, desc, _ in skills:
        skill_text = f"{name}. {desc}" if desc else name
        pairs.append([job_text, skill_text])

    reranker = get_reranker()
    scores = reranker.predict(pairs)  # higher = more relevant

    enriched = []
    for (name, desc, embed_score), rerank_score in zip(skills, scores):
        enriched.append((name, desc, float(embed_score), float(rerank_score)))

    # Sort by rerank_score descending
    enriched.sort(key=lambda x: x[3], reverse=True)
    return enriched
