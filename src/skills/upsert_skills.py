# File: src/skills/upsert_skills.py

import re
import unicodedata
from typing import List, Dict

from src.neo4j_client import get_session
from src.embed_bge import embed_texts


def get_courses_without_skills(limit: int = 100) -> List[Dict]:
    """
    Returns a batch of courses that do NOT yet have any TEACHES->Skill edges.
    This ensures we don't process the same course multiple times.
    """
    query = """
    MATCH (c:Course)
    WHERE NOT (c)-[:TEACHES]->(:Skill)
    RETURN c.course_id      AS course_id,
           c.title_en       AS title_en,
           c.description_en AS description_en
    LIMIT $limit
    """
    with get_session() as session:
        rows = session.run(query, {"limit": limit}).data()
    return rows


# -----------------------------
# Helpers for normalization & dedup
# -----------------------------

def normalize_skill_name(name: str) -> str:
    """
    Normalize a skill name for more robust deduplication.
    - Strip leading/trailing spaces
    - Unicode normalize (NFC)
    - Lowercase (helps with English; Thai unaffected)
    - Collapse multiple spaces into one
    We will reuse this normalized value as the 'name' itself (no extra property).
    """
    if not isinstance(name, str):
        return ""

    text = name.strip()
    if not text:
        return ""

    # Unicode normalize
    text = unicodedata.normalize("NFC", text)

    # Lowercase (for English; no harmful effect on Thai)
    text = text.lower()

    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text)

    return text


def find_existing_skill_by_embedding(
    session,
    embedding: List[float],
    top_k: int = 5,
    threshold: float = 0.85,
) -> Dict:
    """
    Try to find an existing Skill node with similar embedding.
    Uses Neo4j vector index 'skill_embedding_index'.

    Returns:
      {"name": existing_name, "score": similarity} or {}
    """
    query = """
    CALL db.index.vector.queryNodes('skill_embedding_index', $top_k, $embedding)
    YIELD node, score
    RETURN node.name AS name, score
    ORDER BY score DESC
    LIMIT 1
    """
    rec = session.run(query, {"embedding": embedding, "top_k": top_k}).single()

    if not rec:
        return {}

    score = float(rec["score"])
    if score >= threshold:
        return {"name": rec["name"], "score": score}

    return {}


# -----------------------------
# Upsert skills for a single course
# -----------------------------

def upsert_skills_for_course(
    course_id: str,
    course_title: str,
    skills: List[Dict[str, str]],
    embed_threshold: float = 0.85,
) -> None:
    """
    For each extracted skill from a course:
      1) Normalize the skill name (and replace the original name with normalized one).
      2) Embed (name + description) using BGE-m3.
      3) Try to find an existing Skill by embedding similarity.
         - If found (similarity >= threshold): reuse that Skill.
         - Else: create a new Skill node with normalized 'name', description, embedding.
      4) MERGE (Course)-[:TEACHES]->(Skill)

    This:
      - Avoids simple duplicates via normalization.
      - Avoids semantic duplicates via embedding-based KNN.
      - Uses only the 'name' property as the key (no extra slug field).
    """
    if not skills:
        return

    # 1) Normalize skill names in-place & clean descriptions
    cleaned_skills: List[Dict[str, str]] = []
    for s in skills:
        raw_name = (s.get("name") or "").strip()
        if not raw_name:
            continue

        norm_name = normalize_skill_name(raw_name)
        if not norm_name:
            continue

        # Replace original name with normalized one (no extra variable stored in graph)
        s["name"] = norm_name

        desc = (s.get("description") or "").strip()
        s["description"] = desc

        cleaned_skills.append(s)

    if not cleaned_skills:
        return

    # 2) Build texts and compute embeddings
    texts = [
        f"{s['name']} . {s['description']}" if s["description"] else s["name"]
        for s in cleaned_skills
    ]
    embeddings = embed_texts(texts)

    with get_session() as session:
        for s, emb in zip(cleaned_skills, embeddings):
            name = s["name"]              # already normalized
            desc = s["description"]       # cleaned
            embedding = emb

            # 3) Try to reuse an existing Skill by embedding similarity
            existing = find_existing_skill_by_embedding(
                session,
                embedding=embedding,
                top_k=5,
                threshold=embed_threshold,
            )

            if existing:
                # Reuse existing Skill node
                skill_name_to_use = existing["name"]
                # Optionally, you could update description if the existing skill has no description.
                # For now we leave existing as canonical.
            else:
                # No similar skill found -> create new Skill node with normalized 'name'
                session.run(
                    """
                    MERGE (sk:Skill {name: $name})
                    ON CREATE SET sk.description = $description,
                                  sk.embedding  = $embedding
                    """,
                    {
                        "name": name,
                        "description": desc,
                        "embedding": embedding,
                    },
                )
                skill_name_to_use = name

            # 4) Link Course -> Skill
            session.run(
                """
                MATCH (c:Course {course_id: $course_id})
                MATCH (sk:Skill {name: $skill_name})
                MERGE (c)-[:TEACHES]->(sk)
                """,
                {
                    "course_id": course_id,
                    "skill_name": skill_name_to_use,
                },
            )
