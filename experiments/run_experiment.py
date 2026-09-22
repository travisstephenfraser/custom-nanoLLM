"""Run one follow-up experiment: the notebook's own code with one or more settings overridden.

The code executed is the code cells of custom_llm.py (identical to custom_llm.ipynb),
so every variant uses exactly the same data pipeline, training loop and eval runner
as the graded notebooks. Only the named settings change.

Guards against a sweep that silently measures nothing:
  * every override must match its line in the source exactly once, or the run stops;
  * after training, config.json must record the overridden values, or the run stops.
Seed 42 with default settings reproduces the notebook runs; analyze.py checks that.

Usage (from the repository root):
  python experiments/run_experiment.py --name B_seed1 --corpus corpus --seed 1
  python experiments/run_experiment.py --name A_seed1 --corpus none --seed 1
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "runs"
EMPTY = ROOT / "experiments" / "empty_corpus"
DROP = ["corpus.txt", "split.json", "checkpoint.json", "model_untrained.pt"]


def code_cells(source):
    cells, lines, kind = [], [], None
    for line in source.splitlines():
        if line.startswith("# %%"):
            if kind == "code":
                cells.append("\n".join(lines))
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        else:
            lines.append(line)
    if kind == "code":
        cells.append("\n".join(lines))
    return "\n\n".join(cells)


def override(code, pattern, replacement):
    new, count = re.subn(pattern, replacement, code, flags=re.M)
    if count != 1:
        sys.exit(f"Override failed: {pattern!r} matched {count} times (expected 1).")
    return new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--corpus",
        default="corpus",
        help='"corpus", "corpus_split", or "none" (starter only)',
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()

    destination = OUT / args.name
    if destination.exists():
        sys.exit(f"{destination} exists; choose a new --name.")
    folder = args.corpus
    if folder == "none":
        EMPTY.mkdir(parents=True, exist_ok=True)
        folder = str(EMPTY.relative_to(ROOT))

    code = code_cells((ROOT / "custom_llm.py").read_text())
    code = override(code, r'^CORPUS_FOLDER = "corpus"', f'CORPUS_FOLDER = "{folder}"')
    code = override(code, r"^TRAINING_STEPS = 3000\b", f"TRAINING_STEPS = {args.steps}")
    code = override(code, r"^LEARNING_RATE = 0\.001$", f"LEARNING_RATE = {args.lr!r}")
    code = override(
        code,
        r"^SEED, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, BATCH_SIZE = 42, 64, 4, 2, 48, 32$",
        f"SEED, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, BATCH_SIZE = {args.seed}, 64, 4, {args.layers}, 48, 32",
    )

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    namespace = {"__name__": "__experiment__"}
    exec(compile(code, "custom_llm.py", "exec"), namespace)

    run_dir = Path(namespace["run_dir"])
    config = json.loads((run_dir / "config.json").read_text())
    expected = {
        "seed": args.seed,
        "n_layer": args.layers,
        "learning_rate": args.lr,
        "training_steps": args.steps,
    }
    wrong = {k: (config.get(k), v) for k, v in expected.items() if config.get(k) != v}
    if wrong:
        sys.exit(f"config.json does not record the requested settings: {wrong}")

    # Counterfactual probes on the same in-memory model (no weight updates).
    from run_evals import evaluate_suite, load_suite

    evaluate_suite(
        namespace["model"],
        namespace["vocabulary"],
        load_suite(Path("probes/counterfactual_probes.json")),
        run_dir / "probes",
        stage="final",
    )

    for name in DROP:
        (run_dir / name).unlink(missing_ok=True)
    Path(str(run_dir) + ".zip").unlink(missing_ok=True)
    (run_dir / "experiment.json").write_text(
        json.dumps(
            {**vars(args), "corpus_folder": folder, "notebook_run_id": run_dir.name},
            indent=2,
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), destination)
    print("Saved", destination)


if __name__ == "__main__":
    main()
