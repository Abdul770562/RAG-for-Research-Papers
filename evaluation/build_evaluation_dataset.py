import json
from pathlib import Path

from utils.logger import logger


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

QUESTIONS_FILE = (
    BASE_DIR
    / "evaluation"
    / "datasets"
    / "attention_is_all_you_need.json"
)

GROUND_TRUTH_FILE = (
    BASE_DIR
    / "evaluation"
    / "ground_truth"
    / "ground_truth.json"
)

BASELINE_RESULTS_FILE = (
    BASE_DIR
    / "evaluation"
    / "outputs"
    / "baseline_results.json"
)

OUTPUT_FILE = (
    BASE_DIR
    / "evaluation"
    / "outputs"
    / "evaluation_dataset.json"
)


# ==========================================================
# HELPERS
# ==========================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False,
        )


# ==========================================================
# BUILD DATASET
# ==========================================================

def build_dataset():

    logger.info("Loading evaluation artifacts...")

    questions = load_json(QUESTIONS_FILE)
    ground_truth = load_json(GROUND_TRUTH_FILE)
    baseline = load_json(BASELINE_RESULTS_FILE)

    question_map = {q["id"]: q for q in questions}
    gt_map = {g["id"]: g for g in ground_truth}
    baseline_map = {b["id"]: b for b in baseline}

    evaluation_dataset = []

    for question_id in question_map.keys():

        if question_id not in gt_map:
            logger.warning(
                "Missing ground truth for %s",
                question_id,
            )
            continue

        if question_id not in baseline_map:
            logger.warning(
                "Missing baseline result for %s",
                question_id,
            )
            continue

        question = question_map[question_id]
        gt = gt_map[question_id]
        result = baseline_map[question_id]

        sample = {

            "id": question_id,

            "question": question["question"],

            "category": question.get("category"),

            "difficulty": question.get("difficulty"),

            "ground_truth": gt["ground_truth"],

            "answer": result["answer"],

            "contexts": result["contexts"],

            "sources": result["sources"],

            "latency_seconds": result.get(
                "latency_seconds"
            ),
        }

        evaluation_dataset.append(sample)

    save_json(
        evaluation_dataset,
        OUTPUT_FILE,
    )

    logger.info(
        "Evaluation dataset saved to %s",
        OUTPUT_FILE,
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    build_dataset()

