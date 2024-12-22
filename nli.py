import torch
from sentence_transformers import CrossEncoder

if __name__ == "__main__":
    try:
        # Load the model and force the use of the slow tokenizer
        model = CrossEncoder(
            "cross-encoder/nli-deberta-v3-large",
            tokenizer_args={"use_fast": False}
        )
        print("Model loaded successfully.")

        # Ensure the model uses GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model.model.to(device)

        # Define premise and hypothesis
        # TODO: ground truth against statements (answers with degrees of mistakes, can we find where the mistakes are, can i find an evidence of what is written IN THE GROUND TRUTh)
        premise = "For fire to start spreading, it needs three ingredients: heat, fuel, oxygen"
        hypothesis = "For fire to start spreading, it needs three ingredients: fire, wood, oxygen"

        # Predict NLI scores
        scores = model.predict([(premise, hypothesis)])  # Ordering matters
        label_mapping = ['contradiction', 'entailment', 'neutral'] # TODO: check out description of TP, FP, FN and implement thhem according to the logic of nli
        labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]

        # Output the label
        print("Prediction Labels:", labels)
        #TODO: pipeline with autofiles and plots
    except Exception as e:
        print(f"An error occurred: {e}")
