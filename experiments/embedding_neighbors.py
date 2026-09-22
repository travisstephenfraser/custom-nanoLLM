"""Cosine neighbors of chosen words before/after training, and what a 3D map loses.

Reads each run's checkpoint.json (initial and final embedding tables). For each word it
reports the nearest neighbors by cosine similarity over all 64 numbers, and over a 3D
PCA projection of the final table (the kind of map the embedding viewer draws). It also
reports the share of variance the top 3 principal components keep.

Usage:  python experiments/embedding_neighbors.py
"""

import json
from pathlib import Path

import torch

RUNS = {
    "A": ("llm_runs/20260921T190059_023596Z", ["customer", "service", "bank"]),
    "B": (
        "llm_runs/20260921T190646_349052Z",
        ["customer", "above", "left", "blue", "milk"],
    ),
}
K = 5


def neighbors(table, vocab, word, k=K):
    i = vocab.index(word)
    sims = torch.nn.functional.cosine_similarity(table[i : i + 1], table, dim=1)
    sims[i] = float("-inf")
    top = sims.topk(k)
    return [
        (vocab[j], round(s, 3))
        for s, j in zip(top.values.tolist(), top.indices.tolist())
    ]


def main():
    report = {}
    for tag, (run, words) in RUNS.items():
        checkpoint = json.loads(Path(run, "checkpoint.json").read_text())
        vocab = checkpoint["vocabulary"]
        initial = torch.tensor(checkpoint["initial_embeddings"])
        final = torch.tensor(next(iter(checkpoint["weights"].values())))
        if final.shape != initial.shape or torch.equal(final, initial):
            raise SystemExit(
                f"{tag}: final embeddings missing or unchanged; refusing to report neighbors."
            )
        centered = final - final.mean(dim=0)
        _, singular, v = torch.linalg.svd(centered, full_matrices=False)
        variance = singular**2 / (singular**2).sum()
        projected = centered @ v[:3].T
        entry = {
            "run": run,
            "vocabulary_size": len(vocab),
            "variance_kept_by_3_components": round(variance[:3].sum().item(), 3),
            "words": {},
        }
        for word in words:
            full, flat = (
                neighbors(final, vocab, word),
                neighbors(projected, vocab, word),
            )
            entry["words"][word] = {
                "before_64d": neighbors(initial, vocab, word),
                "after_64d": full,
                "after_3d_pca": flat,
                "overlap_3d_vs_64d": len({w for w, _ in full} & {w for w, _ in flat}),
            }
        report[tag] = entry
    out = Path("results/embeddings/neighbors.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    for tag, entry in report.items():
        print(
            f"== {tag}: 3 components keep {entry['variance_kept_by_3_components']:.1%} of the variance"
        )
        for word, row in entry["words"].items():
            print(
                f"  {word}: before {row['before_64d']}\n    after 64D {row['after_64d']}\n"
                f"    after 3D  {row['after_3d_pca']}  (overlap {row['overlap_3d_vs_64d']}/{K})"
            )


if __name__ == "__main__":
    main()
