import torch
from sentence_transformers import CrossEncoder
import pandas as pd
import numpy as np
import re


def parse_log_file(file_path):
    """Parse the cleaned_output.log file to extract predictions grouped by questions."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Split content into prediction blocks
    blocks = content.split("Processing Prediction")
    blocks = [b for b in blocks if b.strip()]  # Remove empty blocks

    predictions_by_question = {}
    questions_text = {}  # Store question text

    for block in blocks:
        try:
            # Extract question number from the block header
            question_match = re.search(r'for Question (\d+)', block)
            if not question_match:
                continue
            question_num = int(question_match.group(1))

            # Extract question text
            question_text_match = re.search(r'Question: (.+?)\n', block)
            if question_text_match and question_num not in questions_text:
                questions_text[question_num] = question_text_match.group(1)

            # Extract statements
            statements = []
            if 'Generated statements:' in block:
                statements_text = block.split('Generated statements:')[1].strip()
                statements = re.findall(r'"([^"]*)"', statements_text)

            # If we found statements, store them
            if statements:
                if question_num not in predictions_by_question:
                    predictions_by_question[question_num] = []
                predictions_by_question[question_num].append(statements)

        except Exception as e:
            print(f"Error processing block: {str(e)}")
            continue

    return predictions_by_question, questions_text


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
    """Split text into sentences using simple rules."""
    sentences = re.split(r'\.(?:\s+|\n+)', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return [s for s in sentences if len(s.split()) > 2]


if __name__ == "__main__":
    try:
        # Load the model
        model = CrossEncoder(
            "cross-encoder/nli-deberta-v3-large",
            tokenizer_args={"use_fast": False}
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.model.to(device)

        # Read CSV file
        df = pd.read_csv('ragas_20_questions_dataset.csv')

        # Read and parse the log file
        predictions_by_question, questions_text = parse_log_file('cleaned_output.log')

        if not predictions_by_question:
            raise ValueError("No predictions were successfully parsed from the log file")

        # Initialize lists for storing results
        csv_results = []  # For storing simplified results
        label_mapping = ['FP (contradiction)', 'TP (entailment)', 'FN (neutral)']

        # Process each row
        for idx, row in df.iterrows():
            premise = row['ground_truth']
            print(f"\nProcessing Question {idx}")

            truth_sentences = segment_text(premise)
            if not truth_sentences:
                truth_sentences = [premise]

            if idx not in predictions_by_question:
                print(f"No predictions found for question {idx}")
                continue

            prediction_groups = predictions_by_question[idx]

            for pred_num, pred_sentences in enumerate(prediction_groups, 1):
                print(f"Processing prediction group {pred_num} with {len(pred_sentences)} statements")

                pairs = []
                for truth_sent in truth_sentences:
                    for pred_sent in pred_sentences:
                        pairs.append((truth_sent, pred_sent))

                scores = model.predict(pairs)
                sentence_labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]

                prediction_f1 = calculate_f1_score(sentence_labels)

                if 'TP (entailment)' in sentence_labels:
                    final_label = 'TP (entailment)'
                elif 'FP (contradiction)' in sentence_labels:
                    final_label = 'FP (contradiction)'
                else:
                    final_label = 'FN (neutral)'

                # Store simplified results for CSV
                csv_results.append({
                    'Question_Number': idx,
                    'Question': questions_text.get(idx, "Question text not found"),
                    'Prediction_Number': pred_num,
                    'F1_Score': prediction_f1
                })

                # Print detailed analysis to terminal
                print(f"\nQuestion {idx + 1}, Prediction {pred_num}:")
                print(f"Label: {final_label}")
                print(f"F1 Score: {prediction_f1:.3f}")
                print("Sentence-level analysis:")
                for i, (pair, label) in enumerate(zip(pairs, sentence_labels)):
                    print(f"Truth: {pair[0]}")
                    print(f"Pred:  {pair[1]}")
                    print(f"Label: {label}\n")

        if csv_results:
            # Create and save simplified DataFrame
            results_df = pd.DataFrame(csv_results)
            results_df.to_csv('evaluation_results.csv', index=False)
            print("\nResults saved to evaluation_results.csv")
        else:
            print("No results were generated to save")

    except Exception as e:
        print(f"An error occurred: {e}")