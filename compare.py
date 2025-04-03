import argparse
import itertools
import pathlib
from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy.stats import spearmanr

from models import AssessmentDataset


def compute_statistic(scores: npt.NDArray, rankings: npt.NDArray) -> Tuple[float, float, float]:
    sp = np.array([spearmanr(s, r).statistic for s, r in zip(scores, rankings)])

    num_bootstraps = 10000
    boot_means = np.array([
        np.mean(np.random.choice(sp, size=len(sp), replace=True))
        for _ in range(num_bootstraps)
    ])
    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])

    return np.mean(sp), ci_lower, ci_upper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=pathlib.Path, required=True)
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        dataset = AssessmentDataset.model_validate_json(f.read())

    ragas_scores = []
    ours_scores = []
    for k, g in itertools.groupby(dataset.entries, lambda i: i.question):
        assessments = list(g)
        assert len(assessments) == 5

        r_scores = []
        o_scores = []
        for e in sorted(assessments, key=lambda a: a.ref_score):
            r_scores.append(e.ragas.score)
            o_scores.append(e.ours.score)

        ragas_scores.append(r_scores)
        ours_scores.append(o_scores)

    true_ranking = np.tile(np.arange(5, 0, -1), (len(dataset.entries), 1))

    print(f"Spearman correlation - Ragas/GT")
    mean, ci_lower, ci_upper = compute_statistic(np.array(ragas_scores), true_ranking)
    print(f"Mean Spearman Correlation: {mean:.3f}")
    print(f"95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print()

    print(f"Spearman correlation - Ours/GT")
    mean, ci_lower, ci_upper = compute_statistic(np.array(ours_scores), true_ranking)
    print(f"Mean Spearman Correlation: {mean:.3f}")
    print(f"95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
