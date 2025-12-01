# File: scripts/run_phase3_jobs_to_skills.py

from src.jobs.map_jobs_to_skills import map_all_jobs_batch

def main():
    # You can tweak these:
    batch_size = 20        # number of jobs per iteration
    top_k = 30             # KNN candidates per job
    top_n = 10             # final NEEDS edges per job
    embed_threshold = 0.80 # minimum embedding-based similarity
    use_reranker = True
    alpha = 0.5            # blending weight between embed_score and rerank_score

    map_all_jobs_batch(
        batch_size=batch_size,
        top_k=top_k,
        top_n=top_n,
        embed_threshold=embed_threshold,
        use_reranker=use_reranker,
        alpha=alpha,
    )

if __name__ == "__main__":
    main()
