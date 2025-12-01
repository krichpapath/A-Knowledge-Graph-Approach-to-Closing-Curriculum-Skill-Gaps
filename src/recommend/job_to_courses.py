# File: src/recommend/job_to_courses.py

from typing import List, Dict
from src.neo4j_client import get_driver
from src.rerank.bge_reranker import rerank_job_skills  # we will reuse reranker for job-course pairs in a moment


def _get_job_text(job_code: str) -> str:
    """
    Build a combined text representation of a job for reranking.
    Uses title, description, tasks (you removed Technology already).
    """
    query = """
    MATCH (j:Job {job_code: $job_code})
    RETURN j.title_en       AS title_en,
           j.description_en AS description_en,
           j.tasks          AS tasks
    """
    driver = get_driver()
    with driver.session() as session:
        rec = session.run(query, {"job_code": job_code}).single()

    if not rec:
        return ""

    title = rec["title_en"] or ""
    desc  = rec["description_en"] or ""
    tasks = rec["tasks"] or ""

    job_text = f"{title}. {desc}. {tasks}".strip()
    return job_text


def recommend_courses_graph(job_code: str, top_n: int = 10) -> List[Dict]:
    """
    Pure knowledge-graph based course recommendation.

    For a given Job:
      - Traverse Job-[:NEEDS]->Skill<-[:TEACHES]-Course
      - Aggregate NEEDS.final_score over shared Skills
      - Compute coverage = matched_skills / total_job_skills
      - graph_score = base_score * coverage

    Returns a list of dicts sorted by graph_score DESC.
    """
    # Cypher:
    # 1) For each course, collect skills it shares with the job, plus NEEDS.final_score.
    # 2) Compute total_job_skills.
    # 3) Compute base_score, coverage, graph_score.
    query = """
    MATCH (j:Job {job_code: $job_code})-[n:NEEDS]->(s:Skill)<-[:TEACHES]-(c:Course)
    WITH j, c,
         collect(s.name) AS skill_names,
         collect(n.final_score) AS final_scores,
         count(*) AS matched_skills
    // get total skills this job needs
    MATCH (j)-[n_all:NEEDS]->(:Skill)
    WITH j, c, skill_names, final_scores, matched_skills, count(n_all) AS total_job_skills
    WITH c, skill_names, matched_skills, total_job_skills,
         reduce(base = 0.0, fs IN final_scores | base + coalesce(fs, 0.0)) AS base_score
    WITH c, skill_names, matched_skills, total_job_skills, base_score,
         CASE
           WHEN total_job_skills > 0 THEN (1.0 * matched_skills) / total_job_skills
           ELSE 0.0
         END AS coverage
    WITH c, skill_names, matched_skills, total_job_skills, base_score, coverage,
         (base_score * coverage) AS graph_score
    RETURN c.course_id      AS course_id,
           c.title_en       AS title_en,
           c.description_en AS description_en,
           skill_names      AS skill_names,
           matched_skills   AS matched_skills,
           total_job_skills AS total_job_skills,
           base_score       AS base_score,
           coverage         AS coverage,
           graph_score      AS graph_score
    ORDER BY graph_score DESC
    LIMIT $top_n
    """
    driver = get_driver()
    with driver.session() as session:
        rows = session.run(query, {"job_code": job_code, "top_n": top_n}).data()

    return rows


def recommend_courses_hybrid(
    job_code: str,
    top_n: int = 10,
    graph_candidates: int = 30,
    w_kg: float = 0.6,
    w_rerank: float = 0.4,
) -> List[Dict]:
    """
    Hybrid recommendation:
      1) Use graph-based scoring to get 'graph_candidates' best courses.
      2) Use reranker (bge-reranker-v2-m3) to score (job_text, course_text).
      3) Combine: final_score = w_kg * graph_score + w_rerank * rerank_score.
      4) Return top_n by final_score.
    """
    # 1) Get graph-based candidates
    graph_recs = recommend_courses_graph(job_code, top_n=graph_candidates)
    if not graph_recs:
        return []

    job_text = _get_job_text(job_code)
    if not job_text:
        # If job not found or no text, just return graph-based
        return graph_recs[:top_n]

    # 2) Prepare pairs for reranker: (job_text, course_text)
    pairs = []
    for r in graph_recs:
        course_text = f"{r['title_en']}. {r['description_en'] or ''}".strip()
        pairs.append([job_text, course_text])

    from sentence_transformers import CrossEncoder  # we can reuse existing reranker wrapper, but let's be explicit
    from src.config import RERANKER_MODEL_NAME

    reranker = CrossEncoder(RERANKER_MODEL_NAME)
    rerank_scores = reranker.predict(pairs)

    # 3) Combine scores
    enriched = []
    for r, rr in zip(graph_recs, rerank_scores):
        graph_score = float(r["graph_score"])
        rerank_score = float(rr)
        final_score = w_kg * graph_score + w_rerank * rerank_score
        item = dict(r)  # copy
        item["rerank_score"] = rerank_score
        item["final_score"] = final_score
        enriched.append(item)

    # Sort by final_score DESC
    enriched.sort(key=lambda x: x["final_score"], reverse=True)

    # 4) Return top_n
    return enriched[:top_n]


def get_job_skill_coverage(job_code: str) -> List[Dict]:
    """
    Helper to inspect which skills for a job are covered by at least one course,
    and which skills appear to have no courses teaching them (gaps).

    Returns: list of {skill_name, num_courses, course_ids}
    """
    query = """
    MATCH (j:Job {job_code: $job_code})-[n:NEEDS]->(s:Skill)
    OPTIONAL MATCH (s)<-[:TEACHES]-(c:Course)
    WITH s, collect(DISTINCT c.course_id) AS course_ids
    RETURN s.name AS skill_name,
           size(course_ids) AS num_courses,
           course_ids
    ORDER BY num_courses ASC, skill_name
    """
    driver = get_driver()
    with driver.session() as session:
        rows = session.run(query, {"job_code": job_code}).data()
    return rows
