# File: src/jobs/map_jobs_to_skills.py

from typing import List, Dict, Optional
from src.neo4j_client import get_session
from src.rerank.bge_reranker import rerank_job_skills


def _get_unmapped_jobs(limit: int = 50) -> List[Dict]:
    """
    Fetch jobs that do NOT yet have any NEEDS->Skill relationships
    AND that are not marked as 'no_skills'.

    This keeps the pipeline idempotent (can run multiple times)
    and avoids retrying jobs that had no skill matches above threshold.
    """
    query = """
    MATCH (j:Job)
    WHERE NOT (j)-[:NEEDS]->(:Skill)
      AND coalesce(j.phase3_status, '') <> 'no_skills'
    RETURN j.job_code AS job_code,
           j.title_en AS title_en,
           j.description_en AS description_en,
           j.technology AS technology,
           j.tasks AS tasks
    LIMIT $limit
    """
    with get_session() as session:
        rows = session.run(query, {"limit": limit}).data()
    return rows


def _get_job_embedding_and_text(job_code: str) -> Optional[Dict]:
    """
    Fetch job embedding and a text representation for reranker.
    """
    query = """
    MATCH (j:Job {job_code: $job_code})
    RETURN j.embedding AS embedding,
           j.title_en AS title_en,
           j.description_en AS description_en,
           j.technology AS technology,
           j.tasks AS tasks
    """
    with get_session() as session:
        rec = session.run(query, {"job_code": job_code}).single()

    if not rec:
        return None

    title = rec["title_en"] or ""
    desc = rec["description_en"] or ""
    tech = rec["technology"] or ""
    tasks = rec["tasks"] or ""

    # A combined text for reranker (query side)
    job_text = f"{title}. {desc}. {tech}. {tasks}".strip()

    return {
        "embedding": rec["embedding"],
        "job_text": job_text,
    }


def _knn_skills_for_job(
    embedding,
    top_k: int = 30,
    embed_threshold: float = 0.80
) -> List[Dict]:
    """
    Use Neo4j vector index to find top-K Skill nodes for a given job embedding.
    Returns list with: name, description, embed_score
    """
    query = """
    CALL db.index.vector.queryNodes('skill_embedding_index', $top_k, $embedding)
    YIELD node, score
    RETURN node.name AS name, node.description AS description, score
    """
    with get_session() as session:
        rows = session.run(query, {"embedding": embedding, "top_k": top_k}).data()

    # Filter by embedding similarity threshold
    filtered = []
    for r in rows:
        s = float(r["score"])
        if s >= embed_threshold:
            filtered.append({
                "name": r["name"],
                "description": r.get("description") or "",
                "embed_score": s,
            })
    return filtered


def _create_needs_edges(
    job_code: str,
    skills: List[Dict]
):
    """
    Create :NEEDS edges from Job to Skills with scores.
    skills: list of dicts containing name, embed_score, rerank_score, final_score
    """
    query = """
    MATCH (j:Job {job_code: $job_code})
    MATCH (s:Skill {name: $skill_name})
    MERGE (j)-[r:NEEDS]->(s)
    SET r.embed_score = $embed_score,
        r.rerank_score = $rerank_score,
        r.final_score = $final_score,
        r.source = 'embed+rerank',
        r.created_at = coalesce(r.created_at, datetime())
    """
    with get_session() as session:
        for s in skills:
            session.run(
                query,
                {
                    "job_code": job_code,
                    "skill_name": s["name"],
                    "embed_score": s["embed_score"],
                    "rerank_score": s["rerank_score"],
                    "final_score": s["final_score"],
                },
            )


def _mark_job_no_skills(job_code: str) -> None:
    """
    Mark a Job as processed in Phase 3 but with no matching skills
    above the similarity threshold.

    This prevents the batch loop from picking it again and lets us
    treat it as a 'skill gap job' later.
    """
    query = """
    MATCH (j:Job {job_code: $job_code})
    SET j.phase3_status = 'no_skills',
        j.phase3_run_at = datetime()
    """
    with get_session() as session:
        session.run(query, {"job_code": job_code})


def map_single_job_to_skills(
    job_code: str,
    top_k: int = 30,
    top_n: int = 10,
    embed_threshold: float = 0.80,
    use_reranker: bool = True,
    alpha: float = 0.5
) -> List[Dict]:
    """
    Map a single Job to Skills:
      1) KNN on embeddings to get candidate Skills.
      2) Optional: rerank candidate Skills with bge-reranker.
      3) Take top_n and create :NEEDS edges.

    If no skills pass the threshold, mark the job as 'no_skills'
    so it won't be retried in batch and can be treated as a gap job.

    Returns list of chosen skills with scores.
    """
    info = _get_job_embedding_and_text(job_code)
    if not info:
        print(f"[WARN] Job {job_code} not found.")
        return []

    embedding = info["embedding"]
    job_text = info["job_text"]

    # 1) KNN using embedding index
    candidates = _knn_skills_for_job(embedding, top_k=top_k, embed_threshold=embed_threshold)
    if not candidates:
        print(f"[INFO] Job {job_code}: no skill candidates above threshold {embed_threshold}")
        _mark_job_no_skills(job_code)
        return []

    # 2) Optional reranking
    if use_reranker:
        # Prepare input for reranker
        basic_list = [(c["name"], c["description"], c["embed_score"]) for c in candidates]
        reranked = rerank_job_skills(job_text, basic_list)  # (name, desc, embed_score, rerank_score)

        # Convert to final format and combine scores if desired
        chosen = []
        for name, desc, embed_score, rerank_score in reranked[:top_n]:
            # Simple fusion: final_score = alpha * embed + (1 - alpha) * rerank
            final_score = alpha * float(embed_score) + (1.0 - alpha) * float(rerank_score)
            chosen.append({
                "name": name,
                "description": desc,
                "embed_score": float(embed_score),
                "rerank_score": float(rerank_score),
                "final_score": float(final_score),
            })
    else:
        # No reranker: just take top_n by embed_score
        candidates.sort(key=lambda x: x["embed_score"], reverse=True)
        chosen = []
        for c in candidates[:top_n]:
            chosen.append({
                "name": c["name"],
                "description": c["description"],
                "embed_score": c["embed_score"],
                "rerank_score": 0.0,
                "final_score": c["embed_score"],
            })

    # 3) Create :NEEDS edges in Neo4j
    _create_needs_edges(job_code, chosen)

    return chosen


def map_all_jobs_batch(
    batch_size: int = 20,
    top_k: int = 30,
    top_n: int = 10,
    embed_threshold: float = 0.80,
    use_reranker: bool = True,
    alpha: float = 0.5
):
    """
    Process jobs in batches: only those that do NOT yet have NEEDS edges
    AND are not marked as 'no_skills'.

    Safe to run multiple times.
    """
    while True:
        jobs = _get_unmapped_jobs(limit=batch_size)
        if not jobs:
            print("No more jobs without NEEDS edges (or all remaining are marked 'no_skills'). Done.")
            break

        print(f"Processing {len(jobs)} jobs in this batch...")
        for j in jobs:
            job_code = j["job_code"]
            print(f"\n[Job] {job_code} - {j['title_en']}")
            chosen_skills = map_single_job_to_skills(
                job_code=job_code,
                top_k=top_k,
                top_n=top_n,
                embed_threshold=embed_threshold,
                use_reranker=use_reranker,
                alpha=alpha,
            )
            print(f"  Mapped to {len(chosen_skills)} skills.")
