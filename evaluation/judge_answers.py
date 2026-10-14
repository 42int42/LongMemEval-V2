#!/usr/bin/env python3
"""Score the LLM-judged questions of an answers file written by run_items.py.

    python evaluation/judge_answers.py <judge model> <answers.jsonl> <questions.jsonl> [--served-as NAME]

The questions run_items.py answered with --defer-llm-eval carry
llm_eval_pending: their eval_function (llm_abstention_checker or
llm_gotchas_checker) asks an LLM judge, which this script calls through the
harness's own score_prediction, with the harness's evaluator settings
(reasoning effort medium, 4,096 completion tokens). Every other question was
scored by the harness already and is copied as it is.

The question text, the reference answer and the eval_function are read from
<questions.jsonl> by id, not from the answers file. The judge talks to the
endpoint the openai client reads from OPENAI_BASE_URL, with the key in
OPENAI_API_KEY, and asks for the model --served-as names (the judge model
itself when it is not given).

Writes <answers.jsonl>.eval-results-<judge model>: every answered question,
with score (1.0 or 0.0), score_bool and judge_model. A question whose judge
reply holds no label is left out and said ("[judge] <id> failed: ..."), not
scored as wrong (the harness itself stops the whole run on one). Beside it,
<answers.jsonl>.aggregated-<judge model>.<domain>.json: the harness's
aggregated metrics for each domain whose questions were all answered here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.harness import (  # noqa: E402
    CATEGORY_MAP,
    LLM_EVAL_FUNCTIONS,
    aggregate_metrics,
    score_prediction,
)
from evaluation.qa_eval_metrics import eval_name  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("judge_model")
    p.add_argument("answers")
    p.add_argument("questions")
    p.add_argument("--served-as", default=None, help="the model name sent to the endpoint")
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium")
    p.add_argument("--max-completion-tokens", type=int, default=4096)
    p.add_argument("--timeout-seconds", type=float, default=600.0)
    args = p.parse_args()

    answers_path = Path(args.answers)
    questions = {q["id"]: q for q in read_jsonl(Path(args.questions))}
    eval_config = {
        "evaluator_model": args.served_as or args.judge_model,
        "evaluator_base_url": None,          # the openai client reads OPENAI_BASE_URL
        "evaluator_api_key": os.getenv("OPENAI_API_KEY"),
        "evaluator_reasoning_effort": args.reasoning_effort,
        "evaluator_max_completion_tokens": args.max_completion_tokens,
        "evaluator_timeout_seconds": args.timeout_seconds,
    }
    out_path = Path(f"{answers_path}.eval-results-{args.judge_model}")
    records: list[dict] = []
    judged = failed = 0
    with out_path.open("w", encoding="utf-8") as out:
        for a in read_jsonl(answers_path):
            q = questions.get(a["id"])
            if q is None:
                sys.exit(f"{a['id']} is not in {args.questions}")
            row = {
                "eval_function": q["eval_function"],
                "eval_name": eval_name(q["eval_function"]),
                "question_item": {"question": q["question"]},
                "answer_gold": q["answer"],
                "response_raw": a["response_raw"],
                "response_parsed_boxed": a["response_parsed_boxed"],
                "is_unknown": a["is_unknown"],
            }
            if a.get("llm_eval_pending"):
                if row["eval_name"] not in LLM_EVAL_FUNCTIONS:
                    sys.exit(f"{a['id']}: pending, but {row['eval_name']} is not an LLM judge")
                try:
                    score_bool, _, _ = score_prediction(row, eval_config)
                except ValueError as exc:            # a reply the harness cannot read as a label
                    print(f"[judge] {a['id']} failed: {exc}", flush=True)
                    failed += 1
                    continue
                judged += 1
            else:
                score_bool = bool(a["score_bool"])
            record = {
                **a,
                "score": 1.0 if score_bool else 0.0,
                "score_bool": score_bool,
                "llm_eval_pending": False,
                "judge_model": args.judge_model if a.get("llm_eval_pending") else None,
            }
            out.write(json.dumps(record, ensure_ascii=True) + "\n")
            out.flush()
            records.append(record)
            print(f"[judge] {a['id']} score={record['score']}", flush=True)

    # The paper's numbers, per domain, when this file holds every question of it.
    for domain in ("web", "enterprise"):
        mine = [r for r in records if r.get("domain") == domain]
        total = sum(1 for q in questions.values() if q.get("domain") == domain)
        if mine and len(mine) == total and all(r.get("category") in CATEGORY_MAP.values() for r in mine):
            agg = aggregate_metrics(mine)
            Path(f"{answers_path}.aggregated-{args.judge_model}.{domain}.json").write_text(
                json.dumps(agg, indent=2) + "\n", encoding="utf-8")
    print(f"[judge] {len(records)} scored, {judged} of them by {eval_config['evaluator_model']}"
          + (f", {failed} left unscored: the judge's reply had no label" if failed else ""), flush=True)


if __name__ == "__main__":
    main()
