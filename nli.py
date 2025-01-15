import torch
from sentence_transformers import CrossEncoder
import pandas as pd
import numpy as np
import pysbd


def calculate_f1_score(labels):
    """Calculate F1 score for a single prediction's sentence-level labels"""
    tp = labels.count('TP (entailment)')
    fp = labels.count('FP (contradiction)')
    fn = labels.count('FN (neutral)')

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def segment_text(text):
    """Split text into sentences using pysbd."""
    seg = pysbd.Segmenter(language="en", clean=True)
    sentences = seg.segment(text)
    # Remove very short segments that might be artifacts
    return [s.strip() for s in sentences if len(s.split()) > 2]


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
        all_results = []
        all_f1_scores = []
        label_mapping = ['FP (contradiction)', 'TP (entailment)', 'FN (neutral)']

        # Process each row
        for idx, row in df.iterrows():
            premise = row['ground_truth']
            predictions = [row[f'prediction_{i}'] for i in range(1, 5)]

            # Split ground truth into sentences using pysbd
            truth_sentences = segment_text(premise)
            if not truth_sentences:
                truth_sentences = [premise]

            # Process each prediction
            row_results = []
            row_f1_scores = []

            for pred_num, prediction in enumerate(predictions, 1):
                # Split prediction into sentences using pysbd
                pred_sentences = segment_text(prediction)
                if not pred_sentences:
                    pred_sentences = [prediction]

                # Create all possible pairs of sentences
                pairs = []
                for truth_sent in truth_sentences:
                    for pred_sent in pred_sentences:
                        pairs.append((truth_sent, pred_sent))

                # Get predictions using the model
                scores = model.predict(pairs)
                sentence_labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]

                # Calculate F1 score for this prediction's sentence pairs
                prediction_f1 = calculate_f1_score(sentence_labels)
                row_f1_scores.append(prediction_f1)

                # Determine final label for this prediction
                if 'TP (entailment)' in sentence_labels:
                    final_label = 'TP (entailment)'
                elif 'FP (contradiction)' in sentence_labels:
                    final_label = 'FP (contradiction)'
                else:
                    final_label = 'FN (neutral)'

                row_results.append(final_label)
                print(f"\nQuestion {idx + 1}, Prediction {pred_num}:")
                print(f"Label: {final_label}")
                print(f"F1 Score: {prediction_f1:.3f}")

                # Print detailed sentence-level analysis
                print("Sentence-level analysis:")
                for i, (pair, label) in enumerate(zip(pairs, sentence_labels)):
                    print(f"Truth: {pair[0]}")
                    print(f"Pred:  {pair[1]}")
                    print(f"Label: {label}\n")

            all_results.extend(row_results)
            all_f1_scores.extend(row_f1_scores)

            # Calculate average F1 score for the question
            avg_f1 = sum(row_f1_scores) / len(row_f1_scores)
            print(f"\nAverage F1 Score for Question {idx + 1}: {avg_f1:.3f}")

        # Create results DataFrame
        results_df = pd.DataFrame({
            'Question_Number': np.repeat(range(1, len(df) + 1), 4),
            'Prediction_Number': np.tile(range(1, 5), len(df)),
            'Label': all_results,
            'F1_Score': all_f1_scores
        })


    except Exception as e:
        print(f"An error occurred: {e}")