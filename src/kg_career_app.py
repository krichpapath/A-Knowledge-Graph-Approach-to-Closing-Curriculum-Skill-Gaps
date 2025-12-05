# File: scripts/kg_career_app.py

import pandas as pd
import gradio as gr

from src.neo4j_client import get_session
from src.recommend.job_to_courses import (
    recommend_courses_graph,
    recommend_courses_hybrid,
    get_job_skill_coverage,
)
from src.recommend.skills_to_jobs import find_jobs_for_skills
from src.recommend.courses_to_jobs import find_jobs_for_courses


# -----------------------------
# Helper: fetch options from KG
# -----------------------------

def fetch_job_options():
    """
    Fetch list of jobs to populate the Job → Courses dropdown.
    Returns labels like: 'OCC_CODE | Job Title'
    """
    query = """
    MATCH (j:Job)
    RETURN j.job_code AS job_code, j.title_en AS title_en
    ORDER BY j.job_code
    """
    with get_session() as session:
        rows = session.run(query).data()

    options = []
    for r in rows:
        code = r["job_code"]
        title = r.get("title_en") or ""
        label = f"{code} | {title}" if title else code
        options.append(label)
    return options


def fetch_skill_options():
    """
    Fetch list of skills to populate the Skills → Jobs multi-select.
    Returns sorted list of skill names.
    """
    query = """
    MATCH (s:Skill)
    RETURN DISTINCT s.name AS name
    ORDER BY name
    """
    with get_session() as session:
        rows = session.run(query).data()

    skills = [r["name"] for r in rows if r["name"]]
    return skills


def fetch_course_options():
    """
    Fetch list of courses to populate the Courses → Jobs multi-select.
    Returns labels like: 'COURSE_ID | Course Title'.
    """
    query = """
    MATCH (c:Course)
    RETURN c.course_id AS course_id, c.title_en AS title_en
    ORDER BY c.course_id
    """
    with get_session() as session:
        rows = session.run(query).data()

    options = []
    for r in rows:
        cid = r["course_id"]
        title = r.get("title_en") or ""
        label = f"{cid} | {title}" if title else cid
        options.append(label)
    return options


def fetch_no_skill_jobs_df() -> pd.DataFrame:
    """
    Fetch jobs that do NOT have any NEEDS->Skill relationships.

    These can be interpreted as 'jobs with no mapped skills' in the
    current KG. They might be:
      - not processed yet by Phase 3, or
      - processed but had no candidates above threshold.
    In both cases, they are interesting as 'skill gap jobs'.
    """
    query = """
    MATCH (j:Job)
    WHERE NOT (j)-[:NEEDS]->(:Skill)
    RETURN j.job_code       AS job_code,
           j.title_en       AS title_en,
           j.description_en AS description_en,
           j.tasks          AS tasks
    ORDER BY j.job_code
    """
    with get_session() as session:
        rows = session.run(query).data()

    if not rows:
        return pd.DataFrame()

    data = []
    for r in rows:
        data.append({
            "job_code": r["job_code"],
            "title_en": r.get("title_en") or "",
            "description_en": r.get("description_en") or "",
            "tasks": r.get("tasks") or "",
        })
    return pd.DataFrame(data)


def run_skill_gap_jobs():
    """
    Gradio callback: returns a DataFrame of jobs that have
    no NEEDS->Skill relationships.
    """
    return fetch_no_skill_jobs_df()


# -----------------------------
# Helper: Phase 4 DataFrames
# -----------------------------

def make_phase4_graph_df(graph_recs):
    """
    Convert graph-only recommendation rows (Job → Courses) into DataFrame.
    """
    if not graph_recs:
        return pd.DataFrame()

    rows = []
    for r in graph_recs:
        rows.append({
            "course_id": r["course_id"],
            "title_en": r["title_en"],
            "graph_score": float(r["graph_score"]),
            "base_score": float(r["base_score"]),
            "coverage": float(r["coverage"]),
            "matched_skills": int(r["matched_skills"]),
            "total_job_skills": int(r["total_job_skills"]),
            "skill_names": ", ".join(r["skill_names"] or []),
        })
    return pd.DataFrame(rows)


def make_phase4_hybrid_df(hybrid_recs):
    """
    Convert hybrid recommendation rows (Job → Courses) into DataFrame.
    """
    if not hybrid_recs:
        return pd.DataFrame()

    rows = []
    for r in hybrid_recs:
        rows.append({
            "course_id": r["course_id"],
            "title_en": r["title_en"],
            "final_score": float(r["final_score"]),
            "graph_score": float(r["graph_score"]),
            "rerank_score": float(r["rerank_score"]),
            "base_score": float(r["base_score"]),
            "coverage": float(r["coverage"]),
            "matched_skills": int(r["matched_skills"]),
            "total_job_skills": int(r["total_job_skills"]),
            "skill_names": ", ".join(r["skill_names"] or []),
        })
    return pd.DataFrame(rows)


def make_phase4_coverage_df(coverage_rows):
    """
    Convert skill coverage rows (per Job) into DataFrame.
    """
    if not coverage_rows:
        return pd.DataFrame()

    rows = []
    for r in coverage_rows:
        num_courses = int(r["num_courses"])
        course_ids = [cid for cid in r["course_ids"] if cid] if r["course_ids"] else []
        rows.append({
            "skill_name": r["skill_name"],
            "num_courses": num_courses,
            "is_gap": (num_courses == 0),
            "course_ids": ", ".join(course_ids),
        })
    return pd.DataFrame(rows)


# -----------------------------
# Helper: Phase 5 & 6 DataFrames
# -----------------------------

def make_jobs_df(job_rows):
    """
    Convert list from find_jobs_for_skills() or find_jobs_for_courses()
    into a standard Jobs DataFrame.
    """
    if not job_rows:
        return pd.DataFrame()

    rows = []
    for r in job_rows:
        rows.append({
            "job_code": r["job_code"],
            "title_en": r["title_en"],
            "graph_score": float(r["graph_score"]),
            "base_score": float(r["base_score"]),
            "coverage_selected": float(r["coverage_selected"]),
            "coverage_job": float(r["coverage_job"]),
            "matched_count": int(r["matched_count"]),
            "total_job_skills": int(r["total_job_skills"]),
            "matched_skills": ", ".join(r["matched_skill_names"]),
            "missing_skills": ", ".join(r["missing_skill_names"]),
        })
    return pd.DataFrame(rows)


# -----------------------------
# Phase 4 callbacks: Job → Courses
# -----------------------------

def run_phase4_recommendations(job_label: str, top_n: int):
    """
    Gradio callback for Phase 4:
      Input: job_label (job_code | title), top_n
      Output: graph-only DF, hybrid DF, coverage DF
    """
    if not job_label:
        empty = pd.DataFrame()
        return empty, empty, empty

    job_code = job_label.split("|")[0].strip()

    # Graph-only
    graph_recs = recommend_courses_graph(job_code, top_n=int(top_n))
    graph_df = make_phase4_graph_df(graph_recs)

    # Hybrid (KG + reranker)
    hybrid_recs = recommend_courses_hybrid(
        job_code=job_code,
        top_n=int(top_n),
        graph_candidates=max(int(top_n) * 3, 30),
        w_kg=0.6,
        w_rerank=0.4,
    )
    hybrid_df = make_phase4_hybrid_df(hybrid_recs)

    # Skill coverage / gaps for this job
    coverage_rows = get_job_skill_coverage(job_code)
    coverage_df = make_phase4_coverage_df(coverage_rows)

    return graph_df, hybrid_df, coverage_df


# -----------------------------
# Phase 5 callbacks: Skills → Jobs
# -----------------------------

def run_phase5_skills_to_jobs(selected_skills: list, top_n_jobs: int):
    """
    Gradio callback for Phase 5:
      Input: list of skill names, top_n_jobs
      Output (3 DFs):
        - Graph-only jobs
        - Hybrid jobs (currently same as graph-only for now)
        - Coverage / gaps view (same DF, but user sees interpretation)
    """
    if not selected_skills:
        empty = pd.DataFrame()
        return empty, empty, empty

    # Pure KG-based ranking of Jobs for Skills
    job_rows = find_jobs_for_skills(selected_skills, top_n=int(top_n_jobs))
    graph_df = make_jobs_df(job_rows)

    # For now, Hybrid = Graph-only (you can later add reranker-based reranking if desired)
    hybrid_df = graph_df.copy()

    # Coverage / gaps: we reuse same DF, since it already contains matched/missing skills per job.
    coverage_df = graph_df.copy()

    return graph_df, hybrid_df, coverage_df


# -----------------------------
# Phase 6 callbacks: Courses → Jobs
# -----------------------------

def run_phase6_courses_to_jobs(selected_courses: list, top_n_jobs: int):
    """
    Gradio callback for Phase 6:
      Input: list of 'COURSE_ID | title' strings, top_n_jobs
      Output (3 DFs):
        - Graph-only jobs
        - Hybrid jobs (currently same as graph-only)
        - Coverage / gaps view (same DF)
    """
    if not selected_courses:
        empty = pd.DataFrame()
        return empty, empty, empty

    # Extract course_ids from 'COURSE_ID | title'
    course_ids = [c.split("|")[0].strip() for c in selected_courses if c]

    # Courses → Jobs via Skills (KG logic)
    job_rows = find_jobs_for_courses(course_ids, top_n_jobs=int(top_n_jobs))
    graph_df = make_jobs_df(job_rows)

    # For now, Hybrid = Graph-only
    hybrid_df = graph_df.copy()

    # Coverage / gaps: reuse same DF
    coverage_df = graph_df.copy()

    return graph_df, hybrid_df, coverage_df


# -----------------------------
# Build Gradio App
# -----------------------------

def build_app():
    job_options = fetch_job_options()
    skill_options = fetch_skill_options()
    course_options = fetch_course_options()

    with gr.Blocks(title="Career Knowledge Graph Explorer") as demo:
        gr.Markdown(
            """
            # 🎓 Career Knowledge Graph Explorer

            This app uses a **Knowledge Graph** of Jobs, Skills, and Courses to support
            curriculum design and student career exploration.

            - **Phase 4 – Job → Courses**:  
              Select a job and see which courses best prepare you, with explainable skill coverage.
            - **Phase 5 – Skills → Jobs**:  
              Select skills you have, and see which jobs fit you and what you are missing.
            - **Phase 6 – Courses → Jobs**:  
              Select courses you have completed, infer your skills, and find matching jobs.
            """
        )

        # ===========================
        # Phase 4: Job → Courses
        # ===========================
        with gr.Tab("Phase 4 – Job → Courses"):
            gr.Markdown(
                """
                ## Phase 4 – From Job to Courses

                1. Choose a **job**.  
                2. The system looks at the skills that job **NEEDS**, and the skills each course **TEACHES**.  
                3. You get ranked courses and a clear explanation of which skills they cover.
                """
            )

            with gr.Row():
                phase4_job_dropdown = gr.Dropdown(
                    choices=job_options,
                    label="Select Job (job_code | title)",
                    interactive=True,
                )
                phase4_topn_slider = gr.Slider(
                    minimum=1,
                    maximum=30,
                    value=10,
                    step=1,
                    label="Top N courses",
                )
                phase4_run_btn = gr.Button("Run Job → Course Recommendations")

            # Sub-tabs inside Phase 4
            with gr.Tab("Graph-only (Knowledge Graph)"):
                gr.Markdown(
                    """
                    **Graph-only ranking** based on Job → NEEDS → Skill ← TEACHES ← Course

                    - `base_score` = sum of NEEDS.final_score for skills shared between the job and the course.  
                    - `coverage`   = fraction of the job's skills that the course covers.  
                    - `graph_score = base_score × coverage`
                    """
                )
                phase4_graph_df = gr.Dataframe(
                    headers=[
                        "course_id",
                        "title_en",
                        "graph_score",
                        "base_score",
                        "coverage",
                        "matched_skills",
                        "total_job_skills",
                        "skill_names",
                    ],
                    label="Graph-only Course Recommendations",
                    interactive=False,
                )

            with gr.Tab("Hybrid (KG + reranker)"):
                gr.Markdown(
                    """
                    **Hybrid ranking** combines:

                    - Knowledge Graph score (`graph_score`), and  
                    - Cross-encoder reranker score (`rerank_score`) from **BAAI/bge-reranker-v2-m3**.

                    `final_score = 0.6 × graph_score + 0.4 × rerank_score`
                    """
                )
                phase4_hybrid_df = gr.Dataframe(
                    headers=[
                        "course_id",
                        "title_en",
                        "final_score",
                        "graph_score",
                        "rerank_score",
                        "base_score",
                        "coverage",
                        "matched_skills",
                        "total_job_skills",
                        "skill_names",
                    ],
                    label="Hybrid KG + Reranker Recommendations",
                    interactive=False,
                )

            with gr.Tab("Skill coverage / gaps"):
                gr.Markdown(
                    """
                    For the selected job, this view shows:

                    - Which **skills** are covered by at least one course.  
                    - Which skills are **gaps** with `is_gap = True` (no course currently teaches them).

                    This is especially useful for curriculum design and identifying **missing content**.
                    """
                )
                phase4_coverage_df = gr.Dataframe(
                    headers=[
                        "skill_name",
                        "num_courses",
                        "is_gap",
                        "course_ids",
                    ],
                    label="Skill Coverage / Gaps for Selected Job",
                    interactive=False,
                )

            phase4_run_btn.click(
                fn=run_phase4_recommendations,
                inputs=[phase4_job_dropdown, phase4_topn_slider],
                outputs=[phase4_graph_df, phase4_hybrid_df, phase4_coverage_df],
            )

        # ===========================
        # Phase 5: Skills → Jobs
        # ===========================
        with gr.Tab("Phase 5 – Skills → Jobs"):
            gr.Markdown(
                """
                ## Phase 5 – From Skills to Jobs

                1. Select **skills** you have (or want to have).  
                2. The system finds Jobs that **NEED** those skills via Job → NEEDS → Skill.  
                3. For each job, you see:
                   - How many of your skills match its needs.  
                   - Which required skills you **already have** and which ones you are **missing**.
                """
            )

            with gr.Row():
                phase5_skills_dropdown = gr.Dropdown(
                    choices=skill_options,
                    label="Select your skills",
                    multiselect=True,
                    interactive=True,
                )
                phase5_topn_jobs_slider = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=10,
                    step=1,
                    label="Top N jobs",
                )
                phase5_run_btn = gr.Button("Find Jobs for Selected Skills")

            with gr.Tab("Graph-only (Knowledge Graph)"):
                gr.Markdown(
                    """
                    Jobs ranked purely by Knowledge Graph signals:

                    - `base_score` = sum of NEEDS.final_score over matched skills.  
                    - `coverage_selected` = matched skills / your selected skills.  
                    - `coverage_job` = matched skills / job's total needed skills.  
                    - `graph_score = base_score × coverage_selected × coverage_job`
                    """
                )
                phase5_graph_df = gr.Dataframe(
                    headers=[
                        "job_code",
                        "title_en",
                        "graph_score",
                        "base_score",
                        "coverage_selected",
                        "coverage_job",
                        "matched_count",
                        "total_job_skills",
                        "matched_skills",
                        "missing_skills",
                    ],
                    label="Jobs (Graph-only ranking)",
                    interactive=False,
                )

            with gr.Tab("Hybrid (KG + reranker)"):
                gr.Markdown(
                    """
                    Hybrid view (currently using the same ranking as Graph-only).  
                    You can later extend this to rerank jobs using a reranker model
                    with a text representation of your skill profile vs job descriptions.
                    """
                )
                phase5_hybrid_df = gr.Dataframe(
                    headers=[
                        "job_code",
                        "title_en",
                        "graph_score",
                        "base_score",
                        "coverage_selected",
                        "coverage_job",
                        "matched_count",
                        "total_job_skills",
                        "matched_skills",
                        "missing_skills",
                    ],
                    label="Jobs (Hybrid placeholder – same as Graph-only)",
                    interactive=False,
                )

            with gr.Tab("Skill coverage / gaps"):
                gr.Markdown(
                    """
                    This view highlights, for each job:

                    - `matched_skills`: which of your selected skills are needed by the job.  
                    - `missing_skills`: which skills the job needs that you do **not** have yet.

                    Use this to see **what you still need to learn** to fully fit each job.
                    """
                )
                phase5_coverage_df = gr.Dataframe(
                    headers=[
                        "job_code",
                        "title_en",
                        "graph_score",
                        "base_score",
                        "coverage_selected",
                        "coverage_job",
                        "matched_count",
                        "total_job_skills",
                        "matched_skills",
                        "missing_skills",
                    ],
                    label="Jobs (Skill coverage / gaps)",
                    interactive=False,
                )

            phase5_run_btn.click(
                fn=run_phase5_skills_to_jobs,
                inputs=[phase5_skills_dropdown, phase5_topn_jobs_slider],
                outputs=[phase5_graph_df, phase5_hybrid_df, phase5_coverage_df],
            )

        # ===========================
        # Phase 6: Courses → Jobs
        # ===========================
        with gr.Tab("Phase 6 – Courses → Jobs"):
            gr.Markdown(
                """
                ## Phase 6 – From Courses to Jobs

                1. Select **courses** you have completed (or plan to complete).  
                2. The system infers your skills via Course → TEACHES → Skill.  
                3. It then finds Jobs that NEED those skills, just like in Phase 5.  
                4. You see which jobs you are close to, and which skills you still need.
                """
            )

            with gr.Row():
                phase6_courses_dropdown = gr.Dropdown(
                    choices=course_options,
                    label="Select your completed courses",
                    multiselect=True,
                    interactive=True,
                )
                phase6_topn_jobs_slider = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=10,
                    step=1,
                    label="Top N jobs",
                )
                phase6_run_btn = gr.Button("Find Jobs for Selected Courses")

            with gr.Tab("Graph-only (Knowledge Graph)"):
                gr.Markdown(
                    """
                    Jobs ranked purely by graph-based reasoning:

                    - Courses → TEACHES → Skills → NEEDS → Jobs  
                    - Scoring is based on the same `graph_score` formula as in Phase 5.
                    """
                )
                phase6_graph_df = gr.Dataframe(
                    headers=[
                        "job_code",
                        "title_en",
                        "graph_score",
                        "base_score",
                        "coverage_selected",
                        "coverage_job",
                        "matched_count",
                        "total_job_skills",
                        "matched_skills",
                        "missing_skills",
                    ],
                    label="Jobs (Graph-only from Courses)",
                    interactive=False,
                )

            with gr.Tab("Hybrid (KG + reranker)"):
                gr.Markdown(
                    """
                    Hybrid view (currently using the same ranking as Graph-only).  
                    Later, you could add free-text descriptions of your learning profile
                    and use a reranker to refine Job ranking.
                    """
                )
                phase6_hybrid_df = gr.Dataframe(
                    headers=[
                        "job_code",
                        "title_en",
                        "graph_score",
                        "base_score",
                        "coverage_selected",
                        "coverage_job",
                        "matched_count",
                        "total_job_skills",
                        "matched_skills",
                        "missing_skills",
                    ],
                    label="Jobs (Hybrid placeholder – same as Graph-only)",
                    interactive=False,
                )

            with gr.Tab("Skill coverage / gaps"):
                gr.Markdown(
                    """
                    This view shows, for each Job matched from your courses:

                    - Which skills are already covered by your selected courses.  
                    - Which job-required skills remain **uncovered** (missing_skills).

                    This can guide you to pick additional courses to close the gap.
                    """
                )
                phase6_coverage_df = gr.Dataframe(
                    headers=[
                        "job_code",
                        "title_en",
                        "graph_score",
                        "base_score",
                        "coverage_selected",
                        "coverage_job",
                        "matched_count",
                        "total_job_skills",
                        "matched_skills",
                        "missing_skills",
                    ],
                    label="Jobs (Skill coverage / gaps from Courses)",
                    interactive=False,
                )

            phase6_run_btn.click(
                fn=run_phase6_courses_to_jobs,
                inputs=[phase6_courses_dropdown, phase6_topn_jobs_slider],
                outputs=[phase6_graph_df, phase6_hybrid_df, phase6_coverage_df],
            )

        # ===========================
        # Skill Gap Jobs: no matched skills in Phase 3
        # ===========================
        with gr.Tab("Skill Gap Jobs"):
            gr.Markdown(
                """
                ## Skill Gap Jobs – No Matched Skills

                These jobs were processed in the Job → Skills mapping step (Phase 3),
                but **no skills passed the similarity threshold**.

                That means:
                - Our current Skills & Courses graph does not provide good coverage
                  for these jobs.
                - They are very interesting for **curriculum designers** and
                  **labor market analysts**, because they may indicate:
                    - Missing or outdated courses
                    - Missing skill concepts in the graph
                    - Or noisy / unusual job descriptions

                You can:
                - Export this table for expert review.
                - Use it to prioritize new skill / course development.
                """
            )

            skill_gap_btn = gr.Button("Load Skill Gap Jobs")
            skill_gap_df = gr.Dataframe(
                headers=[
                    "job_code",
                    "title_en",
                    "description_en",
                    "tasks",
                ],
                label="Jobs with no matched skills (skill_mapping_status = 'no_skills')",
                interactive=False,
            )

            skill_gap_btn.click(
                fn=run_skill_gap_jobs,
                inputs=[],
                outputs=[skill_gap_df],
            )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
