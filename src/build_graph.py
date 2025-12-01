# src/build_graph.py

from src.data_loader import load_all
from src.embed_bge import embed_texts
from src.neo4j_client import get_driver
from src.graph_setup import setup_schema

CREATE_JOB = """
MERGE (j:Job {job_code: $job_code})
SET j.title_en       = $title,
    j.description_en = $description,
    j.tasks          = $tasks,
    j.embedding      = $embedding
"""

CREATE_COURSE = """
MERGE (c:Course {course_id: $course_id})
SET c.title_en       = $title,
    c.description_en = $description,
    c.embedding      = $embedding
"""

def main():
    # 1) Ensure schema is ready
    setup_schema()

    # 2) Load data
    jobs_df, courses_df = load_all()

    # 3) Compute embeddings
    job_embeddings = embed_texts(jobs_df["clean_text"].tolist())
    course_embeddings = embed_texts(courses_df["clean_text"].tolist())

    # 4) Insert into Neo4j
    driver = get_driver()
    with driver.session() as session:
        # Jobs
        for (idx, row), emb in zip(jobs_df.iterrows(), job_embeddings):
            session.run(
                CREATE_JOB,
                {
                    "job_code": row["job_code"],
                    "title": row["Title"],
                    "description": row["Description"],
                    "tasks": row["Tasks"],
                    "embedding": emb,
                },
            )

        # Courses
        for (idx, row), emb in zip(courses_df.iterrows(), course_embeddings):
            session.run(
                CREATE_COURSE,
                {
                    "course_id": row["course_id"],
                    "title": row["COURSE_NAME_EN"],
                    "description": row["COURSE_DESC_EN"],
                    "embedding": emb,
                },
            )

if __name__ == "__main__":
    main()
