import torch
from sentence_transformers import CrossEncoder
import pandas as pd
import numpy as np


def calculate_f1_score(labels):
    tp = labels.count('TP (entailment)')
    fp = labels.count('FP (contradiction)')
    fn = labels.count('FN (neutral)')

    denominator = tp
    numerator = tp + 0.5 * (fp + fn)

    if denominator == 0:
        return 0
    return numerator / denominator

if __name__ == "__main__":
    try:
        # Load the model
        model = CrossEncoder(
            "cross-encoder/nli-deberta-v3-large",
            tokenizer_args={"use_fast": False}
        )

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.model.to(device)

        # Read CSV file
        df = pd.read_csv('ragas_20_questions_dataset.csv')

        # Initialize lists for storing results
        all_pairs = []
        all_results = []
        label_mapping = ['FP (contradiction)', 'TP (entailment)', 'FN (neutral)']

        # Process each row
        for idx, row in df.iterrows():
            premise = row['ground_truth']
            predictions = [row[f'prediction_{i}'] for i in range(1, 5)]

            # Create premise-hypothesis pairs
            pairs = [(premise, pred) for pred in predictions]
            all_pairs.extend(pairs)

            # Get predictions in batches
            scores = model.predict(pairs)
            labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
            all_results.extend(labels)

            # Print results for each row
            print(f"\nQuestion {idx + 1}:")
            for pred_num, label in enumerate(labels, 1):
                print(f"Prediction {pred_num}: {label}")

            f1_score = calculate_f1_score(labels)
            print(f"F1 Score: {f1_score:.3f}")

        # Create results DataFrame
        results_df = pd.DataFrame({
            'Question_Number': np.repeat(range(1, len(df) + 1), 4),
            'Prediction_Number': np.tile(range(1, 5), len(df)),
            'Label': all_results
        })

    except Exception as e:
        print(f"An error occurred: {e}")