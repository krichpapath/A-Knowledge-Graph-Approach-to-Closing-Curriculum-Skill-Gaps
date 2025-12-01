# File: src/recommend/courses_to_jobs.py

from typing import List, Dict
from src.neo4j_client import get_driver
from src.recommend.skills_to_jobs import find_jobs_for_skills


def get_skills_for_courses(selected_course_ids: List[str]) -> List[str]:
    """
    Given a list of course_ids, return the unique skill names
    taught by those courses.

    This uses the knowledge graph pattern:
      (Course)-[:TEACHES]->(Skill)

    We don't weight skills yet; each taught skill is treated equally.
    """
    if not selected_course_ids:
        return []

    query = """
    MATCH (c:Course)-[:TEACHES]->(s:Skill)
    WHERE c.course_id IN $course_ids
    RETURN DISTINCT s.name AS name
    ORDER BY name
    """
    driver = get_driver()
    with driver.session() as session:
        rows = session.run(query, {"course_ids": selected_course_ids}).data()

    skills = [r["name"] for r in rows if r["name"]]
    return skills


def find_jobs_for_courses(
    selected_course_ids: List[str],
    top_n_jobs: int = 10
) -> List[Dict]:
    """
    Courses → Jobs via Skills.

    1) From selected courses, infer a skill set using the KG:
         (Course)-[:TEACHES]->(Skill)

    2) Use Phase 5 logic (skills_to_jobs.find_jobs_for_skills)
       to find jobs that NEED those skills:
         (Skill)<-[:NEEDS]-(Job)

    This takes full advantage of the graph structure:
      - Courses are mapped to skills via TEACHES.
      - Jobs are mapped to skills via NEEDS.
      - Ranking uses NEEDS.final_score and coverage metrics.
    """
    # 1) Courses → Skills
    selected_skills = get_skills_for_courses(selected_course_ids)
    if not selected_skills:
        return []

    # 2) Skills → Jobs (reuse Phase 5 logic)
    jobs = find_jobs_for_skills(selected_skills, top_n=top_n_jobs)
    return jobs
