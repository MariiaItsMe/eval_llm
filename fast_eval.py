from typing import Dict, List, Any

import pysbd
from sentence_transformers import CrossEncoder


def split_into_statements(dataset: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    segmenter = pysbd.Segmenter(language="en", clean=False)

    segmented_dataset = []
    for entry in dataset:
        answer = entry["answer"]
        segments = segmenter.segment(answer)
        segmented_dataset.append({
            "question": entry["question"],
            "ground_truth": entry["ground_truth"],
            "answer": entry["answer"],
            "answer_segments": segments,
        })

    return segmented_dataset


def classification(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

    eval_dataset = []
    for entry in dataset:
        ground_truth = entry["ground_truth"]
        scores = model.predict([(ground_truth, segment) for segment in entry["segments"]])
        label_mapping = ["contradiction", "entailment", "neutral"]
        labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]

        # compute f_score

        eval_dataset.append({
            "question": entry["question"],
            "ground_truth": entry["ground_truth"],
            "answer": entry["answer"],
            "evaluated_segments": [{"segment": segment, "label": label} for segment, label in zip(entry["segments"], labels)],
            "answer_correctness": None, # TODO
        })

        return eval_dataset
