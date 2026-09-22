"""Independent eval-leakage audit for corpus/ (stricter than the notebook's check).

The notebook rejects files that contain an exact eval prompt. That catches copying,
not near-copies. This script re-chunks every corpus file the same way the notebook
does and applies five rules to each passage:

  1. exact_prompt      no normalized eval prompt appears inside the passage
  2. signature         no passage holds 3+ content words from any one eval case
  3. answer_list       no passage holds all four answer choices of any case
  4. banned_pair       the negation answer pairs (red/blue, tea/milk, open/closed)
                       never appear together
  5. subject_pattern   an eval's negation subject (box, door, ava) is never the
                       subject of a negation ("box is not", "ava did not", ...)

It also reports, per eval case, the longest run of consecutive tokens shared with any
passage, so shared sentence frames are visible rather than hidden. A run of 8+
tokens fails the audit.

A leakage check that silently reads nothing reports "clean", so the script refuses
to pass on zero passages and first proves it can catch a planted leak.

Usage:  python check_leakage.py [--corpus corpus] [--report results/leakage/corpus.json]
"""

import argparse
import json
import re
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parent / "evals" / "language_evals.json"
STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "it",
    "of",
    "to",
    "did",
    "not",
    "she",
    "he",
    "that",
    "who",
    "from",
    ".",
    ",",
    "?",
}
BANNED_PAIRS = [{"red", "blue"}, {"tea", "milk"}, {"open", "closed"}]
MAX_SHARED_RUN = 8


def word_tokens(text):
    """Identical to the notebook's tokenizer."""
    return re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", text.lower(), flags=re.UNICODE)


def chunk_text(text, max_tokens=47):
    """Identical to the notebook's chunker: split at [.!?]+whitespace and newlines."""
    chunks = []
    for unit in re.split(r"(?<=[.!?])\s+|\n+", text):
        tokens = word_tokens(unit)
        chunks.extend(
            " ".join(tokens[i : i + max_tokens])
            for i in range(0, len(tokens), max_tokens)
        )
    return chunks


def load_cases(path=SUITE):
    cases = []
    for case in json.loads(Path(path).read_text())["cases"]:
        prompt = word_tokens(case["prompt"])
        signature = {t for t in prompt + [case["answer"]] if t not in STOPWORDS}
        subject = None
        if case["category"] == "negation":
            subject = next(t for t in prompt if t not in STOPWORDS)
        cases.append(
            {
                "id": case["id"],
                "category": case["category"],
                "prompt": prompt,
                "full": prompt + [case["answer"]],
                "signature": signature,
                "choices": set(case["choices"]),
                "subject": subject,
            }
        )
    return cases


def longest_shared_run(a, b):
    best, prev = 0, [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def story_violations(text, cases):
    tokens = word_tokens(text)
    joined, token_set = " " + " ".join(tokens) + " ", set(tokens)
    found = []
    for case in cases:
        if " " + " ".join(case["prompt"]) + " " in joined:
            found.append(("exact_prompt", case["id"]))
        if len(case["signature"] & token_set) >= 3:
            found.append(("signature", case["id"]))
        if case["choices"] <= token_set:
            found.append(("answer_list", case["id"]))
        subject = case["subject"]
        if subject and re.search(rf" {subject} (is|was|did) not ", joined):
            found.append(("subject_pattern", case["id"]))
    for pair in BANNED_PAIRS:
        if pair <= token_set:
            found.append(("banned_pair", "/".join(sorted(pair))))
    return found


def audit(passages, cases):
    violations = []
    for passage in passages:
        for rule, target in story_violations(passage, cases):
            violations.append({"rule": rule, "target": target, "passage": passage})
    runs = {}
    tokenized = [p.split() for p in set(passages)]
    for case in cases:
        best, example = 0, ""
        for tokens in tokenized:
            run = longest_shared_run(case["full"], tokens)
            if run > best:
                best, example = run, " ".join(tokens)
        runs[case["id"]] = {
            "category": case["category"],
            "longest_shared_run": best,
            "prompt_tokens": len(case["prompt"]),
            "example_passage": example,
        }
        if best >= MAX_SHARED_RUN:
            violations.append(
                {"rule": "shared_run", "target": case["id"], "passage": example}
            )
    return violations, runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="corpus")
    parser.add_argument("--report", default="results/leakage/corpus.json")
    args = parser.parse_args()
    cases = load_cases()

    # Known-answer anchor: the audit must flag a planted copy of a real eval prompt.
    planted = " ".join(cases[-1]["full"]) + " ."
    if not any(v["rule"] == "exact_prompt" for v in audit([planted], cases)[0]):
        sys.exit("SELF-TEST FAILED: the audit did not catch a planted eval prompt.")

    files = sorted(
        p
        for p in Path(args.corpus).rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".txt", ".md", ".pdf"}
        and p.name != "README.md"
        and not p.name.startswith(".")
    )
    passages = [c for f in files for c in chunk_text(f.read_text(encoding="utf-8"))]
    if not passages:
        sys.exit(
            f"No passages found in {args.corpus}/; refusing to report a clean audit."
        )

    violations, runs = audit(passages, cases)
    worst = max(runs.values(), key=lambda r: r["longest_shared_run"])
    report = {
        "corpus_files": [str(f) for f in files],
        "passages_scanned": len(passages),
        "unique_passages": len(set(passages)),
        "self_test": "passed",
        "rules": [
            "exact_prompt",
            "signature (>=3 content words of one case)",
            "answer_list",
            "banned_pair",
            "subject_pattern",
            f"shared_run (>= {MAX_SHARED_RUN} tokens)",
        ],
        "violations": violations,
        "longest_shared_run_by_case": runs,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Scanned {len(passages)} passages ({len(set(passages))} unique) from {len(files)} files."
    )
    print(
        f"Longest shared token run with any eval: {worst['longest_shared_run']} "
        f"(limit {MAX_SHARED_RUN}); e.g. {worst['example_passage']!r}"
    )
    print(f"Violations: {len(violations)}. Report: {args.report}")
    sys.exit(1 if violations else 0)


if __name__ == "__main__":
    main()
