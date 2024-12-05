import torch
from sentence_transformers import CrossEncoder

if __name__ == "__main__":
    model = CrossEncoder("cross-encoder/nli-deberta-v3-large")
    model.model.to(torch.device("cuda"))

    premise = "A man is eating pizza."
    hypothesis = "A man walks into the beach"

    scores = model.predict([(premise, hypothesis)])  # ordering matters...
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]

    # Is the hypothesis true, given the premise is the only knowledge about the subject?
    print(labels)


    pass