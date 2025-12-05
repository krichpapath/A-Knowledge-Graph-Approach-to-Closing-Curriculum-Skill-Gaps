# File: src/recommend/skills_to_jobs.py

from typing import List, Dict
from src.neo4j_client import get_session


def find_jobs_for_skills(selected_skills: List[str], top_n: int = 10) -> List[Dict]:
    """
    Given a list of skill names, find jobs that need those skills.

    For each Job j:
      - matched_skill_names: intersection of selected_skills and SkillsNeededBy(j)
      - matched_count      : number of matched skills
      - job_skill_names    : all skills j needs
      - total_job_skills   : len(job_skill_names)
      - base_score         : sum of NEEDS.final_score over matched skills
      - coverage_selected  : matched_count / len(selected_skills)
      - coverage_job       : matched_count / total_job_skills
      - graph_score        : base_score * coverage_selected * coverage_job

    Returns a list of dicts sorted by graph_score DESC.
    """
    if not selected_skills:
        return []

    query = """
    // 1) Find skills that are in the selected_skills list
    MATCH (s:Skill)
    WHERE s.name IN $selected_skills

    // 2) Find jobs that NEED those skills
    MATCH (j:Job)-[n:NEEDS]->(s)
    WITH j,
         collect(DISTINCT s.name) AS matched_skill_names,
         collect(n.final_score)   AS matched_scores,
         count(DISTINCT s)        AS matched_count,
         size($selected_skills)   AS selected_count

    // 3) Get all skills that the job needs
    MATCH (j)-[n_all:NEEDS]->(s_all:Skill)
    WITH j,
         matched_skill_names,
         matched_scores,
         matched_count,
         collect(DISTINCT s_all.name) AS job_skill_names,
         count(DISTINCT s_all)        AS total_job_skills,
         selected_count

    // 4) Compute base_score and coverage metrics
    WITH j,
         matched_skill_names,
         job_skill_names,
         matched_count,
         total_job_skills,
         selected_count,
         reduce(base = 0.0, sc IN matched_scores | base + coalesce(sc, 0.0)) AS base_score,
         CASE
           WHEN selected_count > 0 THEN 1.0 * matched_count / selected_count
           ELSE 0.0
         END AS coverage_selected,
         CASE
           WHEN total_job_skills > 0 THEN 1.0 * matched_count / total_job_skills
           ELSE 0.0
         END AS coverage_job

    // 5) Final graph_score
    WITH j,
         matched_skill_names,
         job_skill_names,
         matched_count,
         total_job_skills,
         base_score,
         coverage_selected,
         coverage_job,
         (base_score * coverage_selected * coverage_job) AS graph_score

    RETURN j.job_code        AS job_code,
           j.title_en        AS title_en,
           matched_skill_names,
           job_skill_names,
           matched_count,
           total_job_skills,
           base_score,
           coverage_selected,
           coverage_job,
           graph_score
    ORDER BY graph_score DESC
    LIMIT $top_n
    """

    with get_session() as session:
        rows = session.run(
            query,
            {"selected_skills": selected_skills, "top_n": top_n}
        ).data()

    # Post-process: compute missing_skills = job_skill_names - matched_skill_names
    results = []
    for r in rows:
        matched = r["matched_skill_names"] or []
        job_skills = r["job_skill_names"] or []
        missing = [s for s in job_skills if s not in matched]

        results.append({
            "job_code": r["job_code"],
            "title_en": r["title_en"] or "",
            "matched_skill_names": matched,
            "job_skill_names": job_skills,
            "missing_skill_names": missing,
            "matched_count": int(r["matched_count"]),
            "total_job_skills": int(r["total_job_skills"]),
            "base_score": float(r["base_score"]),
            "coverage_selected": float(r["coverage_selected"]),
            "coverage_job": float(r["coverage_job"]),
            "graph_score": float(r["graph_score"]),
        })

    return results
