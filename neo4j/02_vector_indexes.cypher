CREATE VECTOR INDEX job_vec_idx IF NOT EXISTS
FOR (j:Job) ON (j.embedding)
OPTIONS { indexConfig: { `vector.dimensions`: 768, `vector.similarity_function`: 'cosine' } };

CREATE VECTOR INDEX course_vec_idx IF NOT EXISTS
FOR (c:Course) ON (c.embedding)
OPTIONS { indexConfig: { `vector.dimensions`: 768, `vector.similarity_function`: 'cosine' } };
