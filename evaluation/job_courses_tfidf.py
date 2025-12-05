# File: scripts/list_courses_tfidf.py
"""
List courses ranked for a given job using TF-IDF cosine similarity.

Usage:
    python scripts/list_courses_tfidf.py              # uses first job in DB
    python scripts/list_courses_tfidf.py 15-1252.00   # specify job_code

This is the classic IR baseline:
- job_text  = title + description + tasks
- course_text = title + description
- score(job, course) = cosine(TF-IDF(job_text), TF-IDF(course_text))
"""

import sys
from typing import Dict, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.neo4j_client import get_session


TOP_N = 20  # how many courses to print


def get_job_text(job_code: str) -> str:
    """
    Build a combined text representation of a job.
    Uses title, description, tasks.
    """
    query = """
    MATCH (j:Job {job_code: $job_code})
    RETURN j.title_en       AS title_en,
           j.description_en AS description_en,
           j.tasks          AS tasks
    """
    with get_session() as session:
        rec = session.run(query, {"job_code": job_code}).single()

    if not rec:
        return ""

    title = rec["title_en"] or ""
    desc = rec["description_en"] or ""
    tasks = rec["tasks"] or ""

    return f"{title}. {desc}. {tasks}".strip()


def get_any_job_code() -> str:
    """
    If no job_code is provided, pick one from the DB.
    """
    query = "MATCH (j:Job) RETURN j.job_code AS job_code LIMIT 1"
    with get_session() as session:
        rec = session.run(query).single()
    return rec["job_code"] if rec else None


def get_all_courses() -> List[Dict]:
    """
    Fetch all courses from Neo4j with course_id, title_en, description_en.
    """
    query = """
    MATCH (c:Course)
    RETURN c.course_id AS course_id,
           c.title_en  AS title_en,
           c.description_en AS description_en
    ORDER BY c.course_id
    """
    with get_session() as session:
        rows = session.run(query).data()
    return rows


def compute_tfidf_scores(job_text: str, courses: List[Dict]) -> Dict[str, float]:
    """
    Compute similarity scores between job_text and each course text using TF-IDF.
    Returns: {course_id: similarity_score}
    """
    course_ids = []
    texts = []
    for c in courses:
        cid = c["course_id"]
        title = c["title_en"] or ""
        desc = c["description_en"] or ""
        text = f"{title}. {desc}".strip()
        course_ids.append(cid)
        texts.append(text)

    # Build TF-IDF matrix for [job_text] + all course texts
    docs = [job_text] + texts
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(docs)  # (1 + N, V)

    job_vec = tfidf[0:1, :]      # (1, V)
    course_vecs = tfidf[1:, :]   # (N, V)

    sims = cosine_similarity(course_vecs, job_vec).reshape(-1)  # (N,)

    scores = {cid: float(sim) for cid, sim in zip(course_ids, sims)}
    return scores


def main():
    # 1) Determine job_code
    if len(sys.argv) > 1:
        job_code = sys.argv[1]
    else:
        job_code = get_any_job_code()
        if not job_code:
            print("No Job nodes found in the database.")
            return
        print(f"No job_code provided. Using example job_code: {job_code}")

    # 2) Get job text
    job_text = get_job_text(job_code)
    if not job_text:
        print(f"Could not find job or job text for job_code={job_code}")
        return

    # 3) Get all courses
    courses = get_all_courses()
    if not courses:
        print("No Course nodes found in the database.")
        return

    # 4) Compute TF-IDF scores
    scores = compute_tfidf_scores(job_text, courses)

    # 5) Sort and print top N
    sorted_courses = sorted(courses, key=lambda c: scores.get(c["course_id"], 0.0), reverse=True)

    print(f"\n=== TF-IDF-based ranking for Job {job_code} ===")
    for i, c in enumerate(sorted_courses[:TOP_N], start=1):
        cid = c["course_id"]
        title = c["title_en"] or ""
        sim = scores.get(cid, 0.0)
        print(f"{i:2d}. {cid} - {title} | sim={sim:.4f}")


if __name__ == "__main__":
    main()
