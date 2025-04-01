from scipy.stats import pearsonr

from nli_ragas_gio import AssessmentDataset

if __name__ == "__main__":
    with open("./assessment.json", "r", encoding="utf-8") as f:
        dataset = AssessmentDataset.model_validate_json(f.read())

    x = [5. - float(e.ref_score[1]) for e in dataset.entries]
    y = [e.ragas.score for e in dataset.entries]
    res = pearsonr(x, y, alternative='two-sided', method=None, axis=0)

    print(f"Pearson correlation - Ragas/GT")
    print(res)
    print(res.confidence_interval(confidence_level=0.95))
    print()

    y = [e.ours.score for e in dataset.entries]
    res = pearsonr(x, y, alternative='two-sided', method=None, axis=0)

    print(f"Pearson correlation - Ours/GT")
    print(res)
    print(res.confidence_interval(confidence_level=0.95))
