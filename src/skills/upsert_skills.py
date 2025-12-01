# File: src/skills/upsert_skills.py

from typing import List, Dict
from src.neo4j_client import get_driver
from src.embed_bge import embed_texts

def get_courses_without_skills(limit: int = 100) -> List[Dict]:
    """
    Returns a batch of courses that do NOT yet have any TEACHES->Skill edges.
    This ensures we don't process the same course multiple times.
    """
    query = """
    MATCH (c:Course)
    WHERE NOT (c)-[:TEACHES]->(:Skill)
    RETURN c.course_id AS course_id,
           c.title_en  AS title_en,
           c.description_en AS description_en
    LIMIT $limit
    """
    driver = get_driver()
    with driver.session() as session:
        rows = session.run(query, {"limit": limit}).data()
    return rows


def upsert_skills_for_course(
    course_id: str,
    course_title: str,
    skills: List[Dict[str, str]]
):
    """
    For each extracted skill:
      - Embed (name + description)
      - MERGE Skill node by name
      - SET description & embedding (on create)
      - MERGE (Course)-[:TEACHES]->(Skill)
    """
    if not skills:
        return

    # Build texts for embedding
    texts = [f"{s['name']} . {s.get('description', '')}" for s in skills]
    embeddings = embed_texts(texts)

    driver = get_driver()
    with driver.session() as session:
        for s, emb in zip(skills, embeddings):
            name = s["name"]
            desc = s.get("description") or ""

            session.run(
                """
                MERGE (sk:Skill {name: $name})
                ON CREATE SET sk.description = $description,
                              sk.embedding  = $embedding
                """,
                {
                    "name": name,
                    "description": desc,
                    "embedding": emb,
                },
            )

            session.run(
                """
                MATCH (c:Course {course_id: $course_id})
                MATCH (sk:Skill {name: $name})
                MERGE (c)-[r:TEACHES]->(sk)
                """,
                {
                    "course_id": course_id,
                    "name": name,
                },
            )
