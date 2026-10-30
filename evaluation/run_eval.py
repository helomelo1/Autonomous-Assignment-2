from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from tqdm import tqdm

from academic_helpdesk.agents import AcademicHelpdesk

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Academic Helpdesk evaluation prompts.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of runs per question (default: 3).")
    parser.add_argument("--output", default="evaluation/results.csv")
    args = parser.parse_args()
    questions = json.loads((ROOT / "evaluation" / "questions.json").read_text(encoding="utf-8"))
    helpdesk = AcademicHelpdesk()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "question_id", "category", "run", "question", "expected_agent", "actual_agent",
            "expected_answerable", "refusal", "answer", "sources", "manual_accuracy",
            "manual_groundedness", "manual_refusal_correct",
        ])
        writer.writeheader()
        for item in tqdm(questions, desc="Evaluating questions"):
            for run in range(1, max(1, args.repeats) + 1):
                result = helpdesk.ask(item["question"])
                writer.writerow({
                    "question_id": item["id"],
                    "category": item["category"],
                    "run": run,
                    "question": item["question"],
                    "expected_agent": item["agent"],
                    "actual_agent": result.get("agent", ""),
                    "expected_answerable": item["answerable"],
                    "refusal": result.get("refusal", False),
                    "answer": result.get("answer", ""),
                    "sources": " | ".join(result.get("sources", [])),
                    "manual_accuracy": "",
                    "manual_groundedness": "",
                    "manual_refusal_correct": "",
                })
    print(f"Wrote {len(questions) * max(1, args.repeats)} answer records to {output}")
    print("Fill the three manual_* columns after checking each answer against its cited source.")


if __name__ == "__main__":
    main()
