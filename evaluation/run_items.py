#!/usr/bin/env python3
"""Answer a given list of questions with one method, for a harness that hands
over its data as files rather than a downloaded dataset directory.

run_eval.py reads a dataset directory (questions.jsonl, trajectories.jsonl,
haystacks/lme_v2_<tier>.json, question_screenshots/) and one domain at a time.
This script takes the files themselves, lays out that directory under
--output-dir, runs run_eval.py once per domain the questions hold, and writes
one answers file with a row per answered question.

- The questions are any subset of questions.jsonl, in its own format.
- The tier is read off the haystack file's name (lme_v2_small.json or
  lme_v2_medium.json).
- trajectories.jsonl is copied with only the trajectories the selected
  questions' haystacks name: the harness reads only those, and the full file
  is 1.2 GB to hold in memory.
- The question screenshots come from exported_data/question_screenshots/ in
  this repository (the 29 files of the dataset at f152293e2355).
- The reader and the LLM judge both talk to the endpoint the openai client
  reads from OPENAI_BASE_URL, with the key in OPENAI_API_KEY. The LLM-judged
  questions are left for evaluation/judge_answers.py (--defer-llm-eval).

Prints one line per question: "[run_items] started <id>" before its domain
runs, then "[run_items] answered <id>", or "[run_items] failed <id>: <reason>"
when the reader gave no text (the harness then records an empty answer, which
the paper scores as wrong; here it is left out of the answers file).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS = REPO_ROOT / "exported_data" / "question_screenshots"
DOMAINS = ("web", "enterprise")
FIRST_ID = re.compile(r'\{\s*"id"\s*:\s*"([^"]+)"')
# What the answers file keeps of a harness record: enough to judge and to read
# an answer, not the prompt and memory context the harness also logs.
KEPT = (
    "question_id",
    "question_type",
    "category",
    "is_abstention_problem",
    "eval_function",
    "question_text",
    "question_image",
    "answer_gold",
    "response_raw",
    "response_parsed_boxed",
    "is_unknown",
    "score",
    "score_bool",
    "llm_eval_pending",
    "memory_context_token_count",
    "memory_context_was_truncated",
    "usage",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tier_of(haystack: Path) -> str:
    m = re.fullmatch(r"lme_v2_(small|medium)\.json", haystack.name)
    if not m:
        sys.exit(f"{haystack.name}: the haystack file is lme_v2_small.json or lme_v2_medium.json")
    return m.group(1)


def keep_trajectories(source: Path, wanted: set[str], target: Path) -> int:
    """Copy the lines of trajectories.jsonl whose id is wanted, reading one line at a time."""
    found = 0
    with source.open(encoding="utf-8") as src, target.open("w", encoding="utf-8") as out:
        for line in src:
            if not line.strip():
                continue
            # the id is the first field of every record: parse only the lines whose id is wanted
            m = FIRST_ID.match(line)
            if m and m.group(1) not in wanted:
                continue
            if json.loads(line).get("id") in wanted:
                out.write(line if line.endswith("\n") else line + "\n")
                found += 1
    return found


def link(target: Path, source: Path) -> None:
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(source.resolve())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--questions", required=True, help="JSONL: the questions to answer, records of questions.jsonl")
    p.add_argument("--haystack", required=True, help="haystacks/lme_v2_<tier>.json")
    p.add_argument("--trajectories", required=True, help="trajectories.jsonl")
    p.add_argument("--method", required=True, help="a run_eval.py --method")
    p.add_argument("--reader-model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--answers-file", required=True, help="JSONL written here, one row per answered question")
    args, extra = p.parse_known_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    questions = read_jsonl(Path(args.questions))
    if not questions:
        sys.exit("no questions given")
    haystack_path = Path(args.haystack)
    tier = tier_of(haystack_path)
    haystack = json.loads(haystack_path.read_text(encoding="utf-8"))
    missing = [q["id"] for q in questions if q["id"] not in haystack]
    if missing:
        sys.exit(f"{len(missing)} questions have no haystack in {haystack_path.name}: {missing[:5]}")

    # The dataset directory run_eval.py reads.
    root = out / "data_root"
    (root / "haystacks").mkdir(parents=True, exist_ok=True)
    with (root / "questions.jsonl").open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=True) + "\n")
    link(root / "haystacks" / haystack_path.name, haystack_path)
    link(root / "question_screenshots", SCREENSHOTS)
    wanted = {t for q in questions for t in haystack[q["id"]]}
    found = keep_trajectories(Path(args.trajectories), wanted, root / "trajectories.jsonl")
    if found != len(wanted):
        sys.exit(f"trajectories.jsonl holds {found} of the {len(wanted)} trajectories the haystacks name")
    print(f"[run_items] {len(questions)} questions, tier {tier}, {found} trajectories", flush=True)

    reader_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("READER_BASE_URL")
    answers = Path(args.answers_file)
    answers.parent.mkdir(parents=True, exist_ok=True)
    answers.write_text("", encoding="utf-8")
    status = 0
    for domain in DOMAINS:
        ids = [q["id"] for q in questions if q.get("domain") == domain]
        if not ids:
            continue
        for qid in ids:
            print(f"[run_items] started {qid}", flush=True)
        run_dir = out / f"{args.method}_{domain}_{tier}"
        cmd = [
            sys.executable, str(REPO_ROOT / "evaluation" / "run_eval.py"),
            "--method", args.method,
            "--data-root", str(root),
            "--domain", domain,
            "--tier", tier,
            "--output-dir", str(run_dir),
            "--question-ids", ",".join(ids),
            "--reader-model", args.reader_model,
            "--defer-llm-eval",
        ]
        if reader_url:
            cmd += ["--reader-base-url", reader_url]
        code = subprocess.call(cmd + extra)
        per_question = run_dir / "per_question.jsonl"
        rows = read_jsonl(per_question) if per_question.exists() else []
        with answers.open("a", encoding="utf-8") as f:
            for r in rows:
                if not r.get("response_raw"):
                    print(f"[run_items] failed {r['question_id']}: the reader gave no text", flush=True)
                    continue
                f.write(json.dumps({"id": r["question_id"], "domain": domain, **{k: r.get(k) for k in KEPT}},
                                   ensure_ascii=True) + "\n")
                print(f"[run_items] answered {r['question_id']}", flush=True)
        if code != 0:
            print(f"[run_items] run_eval.py stopped on the {domain} domain (exit {code})", flush=True)
            status = code
            break
    sys.exit(status)


if __name__ == "__main__":
    main()
