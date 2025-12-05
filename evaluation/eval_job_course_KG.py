# File: scripts/eval_job_course.py
"""
Evaluation script for Job → Course recommendation (KG-only).

Evaluates ONLY the Knowledge Graph-based recommender:
    Job → Skills → Courses (graph_score).

Uses a gold-standard Excel file with columns:
    job_code, job_title, course_id, course_title, relevance_label

Where relevance_label ∈ {"core", "relevant", "irrelevant"}.

We convert labels to numeric grades:
    core       -> 2
    relevant   -> 1
    irrelevant -> 0

Metrics (per job and averaged across jobs):

- Precision@5, Precision@10         (core + relevant treated as "relevant")
- Recall@5, Recall@10               (for all relevant = core+relevant)
- nDCG@5, nDCG@10                   (graded relevance 0/1/2)
- Core@5                            (# of core courses in top5 / 5)
- CoreRecall@10                     (# of core courses in top10 / total core)
"""

import math
from typing import Dict, List

import pandas as pd

from src.recommend.job_to_courses import recommend_courses_graph

# -----------------------------
# Config
# -----------------------------

GOLD_FILE = "data/gold_job_course_labels.xlsx"

# label mapping
LABEL_TO_REL = {
    "core": 2,
    "relevant": 1,
    "irrelevant": 0,
}


# -----------------------------
# Metrics: precision, recall, nDCG
# -----------------------------

def precision_at_k(rel_values: List[int], k: int, rel_threshold: int = 1) -> float:
    """
    rel_values: list of relevance grades in ranked order (predicted ranking).
    Count items with rel >= rel_threshold as relevant (binary).

    Precision@k = (# relevant in top k) / k
    """
    if k <= 0:
        return 0.0
    k = min(k, len(rel_values))
    if k == 0:
        return 0.0

    relevant = sum(1 for r in rel_values[:k] if r >= rel_threshold)
    return relevant / float(k)


def recall_at_k(rel_values: List[int], k: int, rel_threshold: int = 1) -> float:
    """
    Recall@k = (# relevant in top k) / (# relevant in all items)
    rel_values: list of relevance grades in ranked order.
    rel_threshold: threshold to consider something as "relevant", e.g. >=1 for core+relevant.
    """
    if k <= 0:
        return 0.0
    k = min(k, len(rel_values))
    if k == 0:
        return 0.0

    total_relevant = sum(1 for r in rel_values if r >= rel_threshold)
    if total_relevant == 0:
        return 0.0

    relevant_in_top_k = sum(1 for r in rel_values[:k] if r >= rel_threshold)
    return relevant_in_top_k / float(total_relevant)


def dcg_at_k(rel_values: List[int], k: int) -> float:
    """
    Compute DCG@k using graded relevance.
    rel_values: list of int (0,1,2,...) in ranked order.
    """
    k = min(k, len(rel_values))
    if k == 0:
        return 0.0

    dcg = 0.0
    for i in range(k):
        rel = rel_values[i]
        # standard DCG formula: (2^rel - 1) / log2(i+2)
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(rel_values: List[int], k: int) -> float:
    """
    nDCG@k = DCG@k / IDCG@k

    IDCG@k is the DCG@k of the ideal ranking (sorted by rel desc).
    """
    k = min(k, len(rel_values))
    if k == 0:
        return 0.0

    dcg = dcg_at_k(rel_values, k)
    # ideal ranking
    sorted_rels = sorted(rel_values, reverse=True)
    idcg = dcg_at_k(sorted_rels, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def core_counts_at_k(rel_values: List[int], k: int) -> int:
    """
    Return how many core items (rel == 2) are in top k.
    """
    k = min(k, len(rel_values))
    return sum(1 for r in rel_values[:k] if r == 2)


# -----------------------------
# KG scoring
# -----------------------------

def scores_kg(job_code: str, course_ids: List[str]) -> Dict[str, float]:
    """
    KG-based scores using Job → Skills → Courses (graph_score).

    For this job, we:
      - Call recommend_courses_graph(job_code, top_n=large)
      - Get graph_score for connected courses
      - Assign 0.0 for courses with no path (no shared skills)

    Returns: {course_id: score}
    """
    # large top_n to try to cover all relevant courses; KG will only return courses
    # reachable via Job-NEEDS->Skill<-TEACHES-Course
    graph_recs = recommend_courses_graph(job_code, top_n=1000)
    rec_map = {r["course_id"]: float(r["graph_score"]) for r in graph_recs}

    scores = {}
    for cid in course_ids:
        scores[cid] = rec_map.get(cid, 0.0)
    return scores


# -----------------------------
# Evaluation per job
# -----------------------------

def evaluate_for_job(
    job_code: str,
    job_title: str,
    gold_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Evaluate KG-based recommender for a single job.

    gold_df: subset of full gold table filtered to this job_code.

    Returns:
      {
        "P@5": ...,
        "P@10": ...,
        "R@5": ...,
        "R@10": ...,
        "nDCG@5": ...,
        "nDCG@10": ...,
        "Core@5": ...,
        "CoreRecall@10": ...,
      }
    """
    # Build gold relevance map: course_id -> rel_value
    rel_map: Dict[str, int] = {}
    course_ids: List[str] = []
    for _, row in gold_df.iterrows():
        cid = str(row["course_id"])
        label = str(row["relevance_label"]).strip().lower()
        rel = LABEL_TO_REL.get(label, 0)  # default 0 if unknown
        rel_map[cid] = rel
        course_ids.append(cid)

    # Ensure unique course_ids in the order of first appearance
    course_ids = list(dict.fromkeys(course_ids))

    # Gold stats
    gold_rels_all = [rel_map[cid] for cid in course_ids]
    total_core = sum(1 for r in gold_rels_all if r == 2)
    total_relevant = sum(1 for r in gold_rels_all if r >= 1)

    print(f"\n=== Results for Job {job_code} - {job_title} ===")
    print(
        f"Gold stats: #courses={len(course_ids)}, "
        f"#core={total_core}, #core+rel={total_relevant}, "
        f"#irrelevant={len(course_ids) - total_relevant}"
    )

    # --- Get KG scores ---
    kg_scores = scores_kg(job_code, course_ids)

    # --- Produce ranked list and corresponding relevance array ---
    sorted_cids = sorted(course_ids, key=lambda cid: kg_scores.get(cid, 0.0), reverse=True)
    rel_values = [rel_map[cid] for cid in sorted_cids]

    # --- Compute metrics ---
    # Binary relevant = rel >= 1 (core + relevant)
    P5 = precision_at_k(rel_values, 5, rel_threshold=1)
    P10 = precision_at_k(rel_values, 10, rel_threshold=1)
    R5 = recall_at_k(rel_values, 5, rel_threshold=1)
    R10 = recall_at_k(rel_values, 10, rel_threshold=1)
    n5 = ndcg_at_k(rel_values, 5)
    n10 = ndcg_at_k(rel_values, 10)

    # Core-focused metrics
    core_in_top5 = core_counts_at_k(rel_values, 5)
    core_in_top10 = core_counts_at_k(rel_values, 10)
    core_at5 = core_in_top5 / 5.0
    core_recall10 = (core_in_top10 / total_core) if total_core > 0 else 0.0

    metrics = {
        "P@5": P5,
        "P@10": P10,
        "R@5": R5,
        "R@10": R10,
        "nDCG@5": n5,
        "nDCG@10": n10,
        "Core@5": core_at5,
        "CoreRecall@10": core_recall10,
    }

    # Pretty per-job print
    print(f"\n[KG]")
    print(f"  P@5           : {metrics['P@5']:.4f}")
    print(f"  P@10          : {metrics['P@10']:.4f}")
    print(f"  R@5           : {metrics['R@5']:.4f}")
    print(f"  R@10          : {metrics['R@10']:.4f}")
    print(f"  nDCG@5        : {metrics['nDCG@5']:.4f}")
    print(f"  nDCG@10       : {metrics['nDCG@10']:.4f}")
    print(f"  Core@5        : {metrics['Core@5']:.4f}  (#core in top5 / 5)")
    print(f"  CoreRecall@10 : {metrics['CoreRecall@10']:.4f}  (#core in top10 / #core total)")

    return metrics


# -----------------------------
# Main: evaluate all jobs in gold file
# -----------------------------

def main():
    # 1) Load gold file
    df = pd.read_excel(GOLD_FILE)

    # Basic sanity
    required_cols = {"job_code", "job_title", "course_id", "relevance_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Gold file is missing required columns: {missing}")

    # 2) Loop over unique jobs
    all_job_codes = df["job_code"].unique()

    metric_names = ["P@5", "P@10", "R@5", "R@10", "nDCG@5", "nDCG@10", "Core@5", "CoreRecall@10"]

    sum_metrics = {metric: 0.0 for metric in metric_names}
    job_count = 0
    per_job_records = []

    for job_code in all_job_codes:
        subset = df[df["job_code"] == job_code].copy()
        if subset.empty:
            continue

        job_title = str(subset["job_title"].iloc[0])

        metrics = evaluate_for_job(job_code, job_title, subset)
        job_count += 1

        record = {"job_code": job_code, "job_title": job_title}
        for metric_name in metric_names:
            value = metrics[metric_name]
            sum_metrics[metric_name] += value
            record[metric_name] = value
        per_job_records.append(record)

    if job_count == 0:
        print("No jobs found in gold file.")
        return

    # 3) Print averaged results across jobs
    print("\n=== Average KG metrics across jobs ===")
    for metric_name, total_value in sum_metrics.items():
        avg = total_value / job_count
        print(f"  {metric_name}: {avg:.4f}")

    # 4) Per-job metrics table (for inspection/reporting)
    results_df = pd.DataFrame(per_job_records)
    print("\n=== Per-job KG metrics table ===")
    print(results_df)


if __name__ == "__main__":
    main()
