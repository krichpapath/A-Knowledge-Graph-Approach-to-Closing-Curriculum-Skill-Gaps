# File: scripts/eval_job_course.py
"""
Evaluation script for Job → Course recommendation.

Compares 3 methods:

1) KG-based recommender (Job → Skills → Courses)
2) Simple text embedding matcher (BGE-m3)
3) TF-IDF text similarity baseline

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

Also prints macro-averages and relative improvements of KG over EMB.
"""

import math
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.neo4j_client import get_session
from src.embed_bge import embed_texts
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
# Helper: fetch text from Neo4j
# -----------------------------

def get_job_text(job_code: str) -> str:
    """
    Build a combined text representation of a job.
    Uses title, description, tasks.
    """
    query = """
    MATCH (j:Job {job_code: $job_code})
    RETURN j.title_en       AS title_en,
           j.description_en AS description_en,
           j.tasks          AS tasks
    """
    with get_session() as session:
        rec = session.run(query, {"job_code": job_code}).single()

    if not rec:
        return ""

    title = rec["title_en"] or ""
    desc = rec["description_en"] or ""
    tasks = rec["tasks"] or ""

    text = f"{title}. {desc}. {tasks}".strip()
    return text


def get_course_texts(course_ids: List[str]) -> Dict[str, str]:
    """
    For a list of course_ids, fetch title + description from Neo4j
    and return {course_id: text}.
    """
    if not course_ids:
        return {}

    query = """
    MATCH (c:Course)
    WHERE c.course_id IN $course_ids
    RETURN c.course_id AS course_id,
           c.title_en  AS title_en,
           c.description_en AS description_en
    """
    with get_session() as session:
        rows = session.run(query, {"course_ids": course_ids}).data()

    texts = {}
    for r in rows:
        cid = r["course_id"]
        title = r["title_en"] or ""
        desc = r["description_en"] or ""
        text = f"{title}. {desc}".strip()
        texts[cid] = text

    # Default empty text if something is missing (should not happen ideally)
    for cid in course_ids:
        if cid not in texts:
            texts[cid] = ""

    return texts


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
# Scoring methods
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


def scores_embed(job_code: str, course_ids: List[str]) -> Dict[str, float]:
    """
    Simple text embedding matcher baseline (BGE-m3):

    - Build job_text and course_texts
    - Embed with BGE-m3 via embed_texts
    - score(job, course) = cosine(job_vec, course_vec)
    """
    job_text = get_job_text(job_code)
    course_texts = get_course_texts(course_ids)

    texts = [course_texts[cid] for cid in course_ids]

    # Embed
    job_vec = embed_texts([job_text])[0]  # shape (d,)
    course_vecs = embed_texts(texts)      # shape (N, d)

    # Cosine similarity
    job_vec_norm = job_vec / (np.linalg.norm(job_vec) + 1e-9)
    course_norms = course_vecs / (np.linalg.norm(course_vecs, axis=1, keepdims=True) + 1e-9)
    sims = course_norms.dot(job_vec_norm)  # (N,)

    scores = {cid: float(sim) for cid, sim in zip(course_ids, sims)}
    return scores


def scores_tfidf(job_code: str, course_ids: List[str]) -> Dict[str, float]:
    """
    TF-IDF baseline:

    - Build job_text and course_texts
    - Fit TfidfVectorizer on [job_text] + course_texts
    - Compute cosine similarity between job_vec and each course_vec
    """
    job_text = get_job_text(job_code)
    course_texts = get_course_texts(course_ids)

    docs = [job_text] + [course_texts[cid] for cid in course_ids]

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(docs)  # shape (1 + N, V)

    job_vec = tfidf[0:1, :]    # (1, V)
    course_vecs = tfidf[1:, :] # (N, V)

    sims = cosine_similarity(course_vecs, job_vec).reshape(-1)  # (N,)

    scores = {cid: float(sim) for cid, sim in zip(course_ids, sims)}
    return scores


# -----------------------------
# Evaluation per job
# -----------------------------

def evaluate_for_job(
    job_code: str,
    job_title: str,
    gold_df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all 3 methods for a single job.

    gold_df: subset of full gold table filtered to this job_code.

    Returns:
      {
        "KG":    {...metrics...},
        "EMB":   {...metrics...},
        "TFIDF": {...metrics...},
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

    # Ensure unique course_ids in the order of first occurrence
    course_ids = list(dict.fromkeys(course_ids))

    # Gold stats
    gold_rels_all = [rel_map[cid] for cid in course_ids]
    total_core = sum(1 for r in gold_rels_all if r == 2)
    total_relevant = sum(1 for r in gold_rels_all if r >= 1)

    print(f"\n=== Results for Job {job_code} - {job_title} ===")
    print(f"Gold stats: #courses={len(course_ids)}, #core={total_core}, #core+rel={total_relevant}, #irrelevant={len(course_ids) - total_relevant}")

    # --- Get scores for each method ---
    kg_scores = scores_kg(job_code, course_ids)
    emb_scores = scores_embed(job_code, course_ids)
    tfidf_scores = scores_tfidf(job_code, course_ids)

    # --- Produce ranked lists and corresponding relevance arrays ---
    def ranked_rels_from_scores(score_map: Dict[str, float]) -> List[int]:
        sorted_cids = sorted(course_ids, key=lambda cid: score_map.get(cid, 0.0), reverse=True)
        return [rel_map[cid] for cid in sorted_cids]

    rels_kg = ranked_rels_from_scores(kg_scores)
    rels_emb = ranked_rels_from_scores(emb_scores)
    rels_tfidf = ranked_rels_from_scores(tfidf_scores)

    # Metric computation
    def compute_metrics(rel_values: List[int]) -> Dict[str, float]:
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

        return {
            "P@5": P5,
            "P@10": P10,
            "R@5": R5,
            "R@10": R10,
            "nDCG@5": n5,
            "nDCG@10": n10,
            "Core@5": core_at5,
            "CoreRecall@10": core_recall10,
        }

    metrics = {
        "KG": compute_metrics(rels_kg),
        "EMB": compute_metrics(rels_emb),
        "TFIDF": compute_metrics(rels_tfidf),
    }

    # Pretty per-job print
    for method_name, m in metrics.items():
        print(f"\n[{method_name}]")
        print(f"  P@5           : {m['P@5']:.4f}")
        print(f"  P@10          : {m['P@10']:.4f}")
        print(f"  R@5           : {m['R@5']:.4f}")
        print(f"  R@10          : {m['R@10']:.4f}")
        print(f"  nDCG@5        : {m['nDCG@5']:.4f}")
        print(f"  nDCG@10       : {m['nDCG@10']:.4f}")
        print(f"  Core@5        : {m['Core@5']:.4f}  (#core in top5 / 5)")
        print(f"  CoreRecall@10 : {m['CoreRecall@10']:.4f}  (#core in top10 / #core total)")

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

    # Accumulate metrics to compute averages
    metric_names = ["P@5", "P@10", "R@5", "R@10", "nDCG@5", "nDCG@10", "Core@5", "CoreRecall@10"]
    methods = ["KG", "EMB", "TFIDF"]

    sum_metrics = {
        method: {metric: 0.0 for metric in metric_names}
        for method in methods
    }
    job_count = 0

    per_job_records = []  # for optional pandas summary table

    for job_code in all_job_codes:
        subset = df[df["job_code"] == job_code].copy()
        if subset.empty:
            continue

        job_title = str(subset["job_title"].iloc[0])

        metrics = evaluate_for_job(job_code, job_title, subset)
        job_count += 1

        # accumulate
        for method_name in methods:
            rec = {"job_code": job_code, "job_title": job_title, "method": method_name}
            for metric_name in metric_names:
                value = metrics[method_name][metric_name]
                sum_metrics[method_name][metric_name] += value
                rec[metric_name] = value
            per_job_records.append(rec)

    if job_count == 0:
        print("No jobs found in gold file.")
        return

    # 3) Print averaged results across jobs
    print("\n=== Average metrics across jobs ===")
    for method_name in methods:
        print(f"\n[{method_name}]")
        for metric_name in metric_names:
            avg = sum_metrics[method_name][metric_name] / job_count
            print(f"  {metric_name}: {avg:.4f}")

    # 4) Build a DataFrame of per-job metrics (optional but very useful)
    results_df = pd.DataFrame(per_job_records)
    print("\n=== Per-job metrics table ===")
    print(results_df)

    # 5) Compute macro-average table for easier comparison
    avg_rows = []
    for method_name in methods:
        row = {"method": method_name}
        for metric_name in metric_names:
            row[metric_name] = sum_metrics[method_name][metric_name] / job_count
        avg_rows.append(row)
    avg_df = pd.DataFrame(avg_rows)
    print("\n=== Macro-average metrics ===")
    print(avg_df)

    # 6) Show relative improvement of KG over EMB for key metrics
    print("\n=== Relative improvement of KG over EMB (macro-average) ===")
    key_metrics = ["P@5", "P@10", "nDCG@5", "nDCG@10", "Core@5", "CoreRecall@10"]
    emb_row = avg_df[avg_df["method"] == "EMB"].iloc[0]
    kg_row = avg_df[avg_df["method"] == "KG"].iloc[0]

    for metric_name in key_metrics:
        base = float(emb_row[metric_name])
        kg = float(kg_row[metric_name])
        if base == 0:
            print(f"  {metric_name}: EMB=0.0000, KG={kg:.4f} (cannot compute % improvement)")
        else:
            improvement = (kg - base) / base * 100.0
            print(f"  {metric_name}: EMB={base:.4f}, KG={kg:.4f}, Δ={kg - base:+.4f} ({improvement:+.2f}% )")


if __name__ == "__main__":
    main()
