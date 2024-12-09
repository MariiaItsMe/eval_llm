import torch
from sentence_transformers import CrossEncoder

if __name__ == "__main__":
    model = CrossEncoder("cross-encoder/nli-deberta-v3-large", tokenizer_args=dict(use_fast=False))
    model.model.to(torch.device("cuda"))

    scores = model.predict([("Albert Einstein was a German-theoretical physicist.",
                             "Albert Einstein is mainly known by the theory of relativity."),

                            ("Albert Einstein was a German-theoretical physicist.",
                             "Albert Einstein is a physicist."),

                            ("Albert Einstein was a German-theoretical physicist.",
                             "Albert Einstein was American."),

                            # premise, hypothesis (ordering matters)
                            ("All dogs are mammals.", "This dog is a mammal."),  # ==> entailment
                            ("This dog is a mammal.", "All dogs are mammals."),  # ==> neutral

                            ("To start and propagate, a fire needs three elements: heat, fuel, and oxygen. Eliminating "
                             "any of these three elements will effectively stop a fire. One can remove the fuel by "
                             "cleaning the area with firebreaks (spaces cleared from vegetation), or by starting a "
                             "controlled fire. Another way to stop a fire is by removing heat, typically done by "
                             "pouring water and particularly effective for solid combustibles like wood or paper. "
                             "The last way to stop a fire is by removing oxygen. This approach typically involve "
                             "throwing at the fire materials like sand, dirt, or fire blankets.",
                             "Removing heat, fuel or oxygen can effectively extinguish a fire."),
                            ])
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]

    # Is the hypothesis true, given the premise is the only knowledge about the subject?
    print(labels)

    pass
