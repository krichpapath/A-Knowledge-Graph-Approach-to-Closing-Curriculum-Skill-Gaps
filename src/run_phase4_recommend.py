# File: scripts/run_phase4_recommend.py

import sys
from src.recommend.job_to_courses import (
    recommend_courses_graph,
    recommend_courses_hybrid,
    get_job_skill_coverage,
)
from src.neo4j_client import get_session


def _get_any_job_code() -> str:
    """
    Helper: pick one job_code from the DB if none is supplied.
    """
    query = "MATCH (j:Job) RETURN j.job_code AS job_code LIMIT 1"
    with get_session() as session:
        rec = session.run(query).single()
    return rec["job_code"] if rec else None


def main():
    if len(sys.argv) > 1:
        job_code = sys.argv[1]
    else:
        job_code = _get_any_job_code()
        if not job_code:
            print("No Job nodes found in the database.")
            return
        print(f"No job_code provided. Using example job_code: {job_code}")

    print(f"\n=== Graph-only recommendations for Job {job_code} ===")
    graph_recs = recommend_courses_graph(job_code, top_n=10)
    if not graph_recs:
        print("No recommendations found. Check if this job has NEEDS edges and courses have TEACHES edges.")
    else:
        for i, r in enumerate(graph_recs, start=1):
            print(f"\n#{i} Course {r['course_id']} - {r['title_en']}")
            print(f"  graph_score   : {r['graph_score']:.4f}")
            print(f"  matched_skills: {r['matched_skills']} / {r['total_job_skills']}")
            print(f"  skills        : {', '.join(r['skill_names'])}")

    print(f"\n=== Hybrid (KG + reranker) recommendations for Job {job_code} ===")
    hybrid_recs = recommend_courses_hybrid(job_code, top_n=10, graph_candidates=30, w_kg=0.6, w_rerank=0.4)
    if not hybrid_recs:
        print("No recommendations (hybrid).")
    else:
        for i, r in enumerate(hybrid_recs, start=1):
            print(f"\n#{i} Course {r['course_id']} - {r['title_en']}")
            print(f"  final_score   : {r['final_score']:.4f}")
            print(f"  graph_score   : {r['graph_score']:.4f}")
            print(f"  rerank_score  : {r['rerank_score']:.4f}")
            print(f"  matched_skills: {r['matched_skills']} / {r['total_job_skills']}")
            print(f"  skills        : {', '.join(r['skill_names'])}")

    print(f"\n=== Skill coverage for Job {job_code} ===")
    coverage = get_job_skill_coverage(job_code)
    for row in coverage:
        print(f"Skill: {row['skill_name']}")
        print(f"  num_courses: {row['num_courses']}")
        if row["num_courses"] == 0:
            print("  -> GAP (no course teaches this skill)")
        else:
            print(f"  courses   : {', '.join([cid for cid in row['course_ids'] if cid])}")
        print("")

if __name__ == "__main__":
    main()
