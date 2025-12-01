# src/data_loader.py

import pandas as pd
from typing import Tuple
from src.config import JOBS_FILE, COURSES_FILE

def load_jobs() -> pd.DataFrame:
    """
    Load jobs data with header:
    O*NET-SOC Code, Title, Description, Tasks
    """
    if JOBS_FILE.lower().endswith(".xlsx"):
        df = pd.read_excel(JOBS_FILE)
    else:
        df = pd.read_csv(JOBS_FILE)

    # Normalize column names (strip spaces)
    df = df.rename(columns=lambda c: c.strip())

    required = ["O*NET-SOC Code", "Title", "Description", "Tasks"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column in jobs file: {col}")

    # Build a combined text for embedding
    df["job_code"] = df["O*NET-SOC Code"].astype(str)

    # Fill NaNs
    for col in required[1:]:
        df[col] = df[col].fillna("")

    df["clean_text"] = (
        "Job Title: " + df["Title"].astype(str) + " . " +
        "Description: " + df["Description"].astype(str) + " . " +
        "Tasks: " + df["Tasks"].astype(str)
    )

    return df[["job_code", "Title", "Description", "Tasks", "clean_text"]]


def load_courses() -> pd.DataFrame:
    """
    Load courses data with header:
    COURSE_ID, COURSE_NAME_EN, COURSE_DESC_EN
    """
    if COURSES_FILE.lower().endswith(".xlsx"):
        df = pd.read_excel(COURSES_FILE)
    else:
        df = pd.read_csv(COURSES_FILE)

    df = df.rename(columns=lambda c: c.strip())

    required = ["COURSE_ID", "COURSE_NAME_EN", "COURSE_DESC_EN"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column in courses file: {col}")

    df["course_id"] = df["COURSE_ID"].astype(str)
    df["COURSE_NAME_EN"] = df["COURSE_NAME_EN"].fillna("")
    df["COURSE_DESC_EN"] = df["COURSE_DESC_EN"].fillna("")

    df["clean_text"] = (
        df["COURSE_NAME_EN"].astype(str) + " . " +
        df["COURSE_DESC_EN"].astype(str)
    )

    return df[["course_id", "COURSE_NAME_EN", "COURSE_DESC_EN", "clean_text"]]


def load_all() -> Tuple[pd.DataFrame, pd.DataFrame]:
    jobs = load_jobs()
    courses = load_courses()
    return jobs, courses
