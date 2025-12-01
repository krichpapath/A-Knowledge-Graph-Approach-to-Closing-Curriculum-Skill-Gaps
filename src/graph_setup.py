# File: src/graph_setup.py

from src.neo4j_client import get_driver
from src.config import EMBEDDING_DIM

CONSTRAINTS_CYPHER = [
    # Unique job_code for Job
    """
    CREATE CONSTRAINT job_code_unique IF NOT EXISTS
    FOR (j:Job)
    REQUIRE j.job_code IS UNIQUE
    """,
    # Unique course_id for Course
    """
    CREATE CONSTRAINT course_id_unique IF NOT EXISTS
    FOR (c:Course)
    REQUIRE c.course_id IS UNIQUE
    """,
    # Unique name for Skill
    """
    CREATE CONSTRAINT skill_name_unique IF NOT EXISTS
    FOR (s:Skill)
    REQUIRE s.name IS UNIQUE
    """
]

VECTOR_INDEXES_CYPHER = [
    # Job embedding index
    f"""
    CREATE VECTOR INDEX job_embedding_index IF NOT EXISTS
    FOR (j:Job) ON (j.embedding)
    OPTIONS {{
      indexConfig: {{
        `vector.dimensions`: {EMBEDDING_DIM},
        `vector.similarity_function`: 'cosine'
      }}
    }}
    """,
    # Course embedding index
    f"""
    CREATE VECTOR INDEX course_embedding_index IF NOT EXISTS
    FOR (c:Course) ON (c.embedding)
    OPTIONS {{
      indexConfig: {{
        `vector.dimensions`: {EMBEDDING_DIM},
        `vector.similarity_function`: 'cosine'
      }}
    }}
    """,
    # Skill embedding index
    f"""
    CREATE VECTOR INDEX skill_embedding_index IF NOT EXISTS
    FOR (s:Skill) ON (s.embedding)
    OPTIONS {{
      indexConfig: {{
        `vector.dimensions`: {EMBEDDING_DIM},
        `vector.similarity_function`: 'cosine'
      }}
    }}
    """
]

def setup_schema():
    driver = get_driver()
    with driver.session() as session:
        for q in CONSTRAINTS_CYPHER:
            session.run(q)
        for q in VECTOR_INDEXES_CYPHER:
            session.run(q)
