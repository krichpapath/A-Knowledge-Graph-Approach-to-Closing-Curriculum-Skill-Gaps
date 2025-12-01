# File: src/skills/run_phase2_skills.py

from src.skills.upsert_skills import get_courses_without_skills, upsert_skills_for_course
from src.skills.extract_skills import extract_skills_for_course

LLM_MODEL = "llama3.1:8b"

def process_batch(batch_size: int = 20):
    courses = get_courses_without_skills(limit=batch_size)
    if not courses:
        print("No more courses without skills. All done.")
        return False
        
    print(f"Processing {len(courses)} courses...")
    for row in courses:
        cid = row["course_id"]
        title = row["title_en"] or ""
        desc  = row["description_en"] or ""

        print(f"\n[Course] {cid} - {title}")
        skills = extract_skills_for_course(LLM_MODEL, title, desc)
        print(f"  Extracted {len(skills)} skills")

        upsert_skills_for_course(cid, title, skills)

    return True

def main():
    # Loop until no more unprocessed courses
    while True:
        has_more = process_batch(batch_size=10)
        if not has_more:
            break

if __name__ == "__main__":
    main()
