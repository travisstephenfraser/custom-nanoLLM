"""Aggregate experiments/runs/* into experiments/summary.csv and experiments/summary.md.

Checks before summarizing (a sweep that silently ran one configuration would look like
"settings don't matter"):
  * every run's config matches its experiment.json request;
  * no two runs produced the same trained model (identical hashes mean an override failed);
  * the three seed-42 default runs reproduce the graded notebook runs case for case.

Usage:  python experiments/analyze.py
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

RUNS = Path("experiments/runs")
NOTEBOOK_RUNS = {
    "A_seed42": "20260921T190059_023596Z",
    "B_seed42": "20260921T190646_349052Z",
    "C_seed42": "20260921T190700_291475Z",
}
TARGETED = ("negation", "spatial_relations")


def load(path):
    return json.loads(Path(path).read_text())


def results(path):
    data = load(path)
    return {
        row["id"]: row for row in (data if isinstance(data, list) else data["results"])
    }


def margin(row):
    """Probability of the expected word minus the best wrong choice (negative = wrong)."""
    probs = row.get("choice_probabilities") or {}
    if not probs:
        return None
    wrong = max(p for word, p in probs.items() if word != row["expected"])
    return probs[row["expected"]] - wrong


def summarize(run):
    request, config = load(run / "experiment.json"), load(run / "config.json")
    for key, field in [
        ("seed", "seed"),
        ("layers", "n_layer"),
        ("lr", "learning_rate"),
        ("steps", "training_steps"),
    ]:
        if config[field] != request[key]:
            raise SystemExit(
                f"{run.name}: config {field}={config[field]} but requested {request[key]}"
            )
    final = load(run / "language_evals/final/eval_summary.json")
    rows = results(run / "language_evals/final/eval_results.json")
    probes = results(run / "probes/eval_results.json")
    history = load(run / "history.json")
    recency_file = run / "recency_probes/eval_results.json"
    recency = results(recency_file) if recency_file.exists() else None
    cats = final["by_category"]
    margins = {
        c: [margin(r) for r in rows.values() if r["category"] == c] for c in TARGETED
    }
    condition = re.sub(r"_seed\d+$", "", run.name)
    return {
        "run": run.name,
        "condition": condition,
        "seed": config["seed"],
        "corpus": request["corpus_folder"],
        "layers": config["n_layer"],
        "lr": config["learning_rate"],
        "steps": config["training_steps"],
        "parameters": config["parameters"],
        "vocabulary": config["vocabulary_size"],
        "final_train_loss": round(history[-1]["training_loss"], 4),
        "final_val_loss": round(history[-1]["validation_loss"], 4),
        "correct_48": final["overall"]["correct"],
        "scorable_48": final["overall"]["scorable"],
        "starter_patterns_16": final["by_group"]["starter_patterns"]["correct"],
        "transfer_8": final["by_group"]["starter_transfer"]["correct"],
        "negation_3": cats["negation"]["correct"],
        "spatial_3": cats["spatial_relations"]["correct"],
        "negation_margin": round(sum(margins["negation"]) / 3, 3)
        if None not in margins["negation"]
        else None,
        "spatial_margin": round(sum(margins["spatial_relations"]) / 3, 3)
        if None not in margins["spatial_relations"]
        else None,
        "probe_flip_6": sum(
            r["score"] for r in probes.values() if r["group"] == "flip"
        ),
        "probe_fresh_6": sum(
            r["score"] for r in probes.values() if r["group"] == "fresh"
        ),
        "recency_trap_6": sum(r["score"] for r in recency.values()) if recency else None,
        "recency_picked_negated": sum(r["predicted_choice"] == r["choices"][1] for r in recency.values()) if recency else None,
        "model_sha256": final["model_sha256"],
    }


def main():
    rows = [
        summarize(run)
        for run in sorted(RUNS.iterdir())
        if (run / "experiment.json").exists()
    ]
    if not rows:
        raise SystemExit("No runs found.")
    hashes = defaultdict(list)
    for row in rows:
        hashes[row["model_sha256"]].append(row["run"])
    duplicates = [names for names in hashes.values() if len(names) > 1]
    if duplicates:
        raise SystemExit(
            f"Identical trained models (an override did not apply): {duplicates}"
        )
    for name, notebook_run in NOTEBOOK_RUNS.items():
        mine = results(RUNS / name / "language_evals/final/eval_results.json")
        graded = results(
            Path("llm_runs") / notebook_run / "language_evals/final/eval_results.json"
        )
        same = all(
            (mine[i]["score"], mine[i]["generated_text"])
            == (graded[i]["score"], graded[i]["generated_text"])
            for i in graded
        )
        if not same:
            raise SystemExit(f"{name} does not reproduce notebook run {notebook_run}.")

    with open("experiments/summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    order = [
        "A",
        "B",
        "C",
        "B_layers1",
        "B_layers4",
        "B_lr0.0001",
        "B_lr0.01",
        "B_lr0.03",
        "B_steps10000",
    ]
    cols = [
        ("correct_48", "Correct /48"),
        ("transfer_8", "Transfer /8"),
        ("negation_3", "Negation /3"),
        ("spatial_3", "Spatial /3"),
        ("negation_margin", "Negation margin"),
        ("spatial_margin", "Spatial margin"),
        ("probe_flip_6", "Flip probes /6"),
        ("probe_fresh_6", "Fresh probes /6"),
        ("recency_trap_6", "Recency traps /6"),
        ("final_val_loss", "Val loss"),
    ]
    lines = [
        "| Condition | Seeds | " + " | ".join(label for _, label in cols) + " |",
        "|---|---|" + "---:|" * len(cols),
    ]
    for condition in order:
        group = sorted(
            by_condition.get(condition, []),
            key=lambda r: [42, 1, 2, 3].index(r["seed"]),
        )
        if not group:
            continue
        cells = []
        for key, _ in cols:
            values = [r[key] for r in group]
            if all(v is None for v in values):
                cells.append("n/a")
            elif isinstance(values[0], float):
                cells.append(
                    " / ".join(f"{v:.2f}" if v is not None else "n/a" for v in values)
                )
            else:
                cells.append(" / ".join(str(v) for v in values))
        lines.append(
            f"| {condition} | {', '.join(str(r['seed']) for r in group)} | "
            + " | ".join(cells)
            + " |"
        )
    Path("experiments/summary.md").write_text(
        "Per-seed values, in seed order 42 / 1 / 2 / 3. Margin = p(correct) - p(best wrong choice), "
        "averaged over the 3 cases.\n\n" + "\n".join(lines) + "\n"
    )
    print(Path("experiments/summary.md").read_text())
    print(f"{len(rows)} runs; all models unique; seed-42 runs reproduce the notebooks.")


if __name__ == "__main__":
    main()
