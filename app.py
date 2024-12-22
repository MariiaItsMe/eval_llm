import os
import re
import time
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas.metrics import answer_correctness
from ragas import evaluate

# Load environment variables
load_dotenv()


def evaluate_all_predictions(csv_path):
    """
    Evaluate answer correctness using Ragas metrics for all predictions in a given CSV dataset
    """
    df = pd.read_csv(csv_path)

    # Extract questions, ground truths, and predictions
    questions = df['question'].tolist()
    ground_truths = df['ground_truth'].tolist()
    predictions = [
        df[f'prediction_{i}'].tolist() for i in range(1, 5)
    ]

    # Prepare results DataFrame
    results_list = []

    # Evaluate each prediction
    for pred_idx, prediction in enumerate(predictions, 1):
        # Prepare dataset for this prediction
        data_samples = {
            'question': questions,
            'ground_truth': ground_truths,
            'response': prediction
        }
        dataset = Dataset.from_dict(data_samples)

        time.sleep(1)  # 1-second pause between operations

        # Evaluate using Ragas
        score = evaluate(dataset, metrics=[answer_correctness])

        # Convert to pandas
        results_df = score.to_pandas()


        # Add additional columns
        results_df['prediction_number'] = pred_idx
        results_df['question'] = questions
        results_df['ground_truth'] = ground_truths
        results_df['prediction'] = prediction

        results_list.append(results_df)

    # Concatenate results from all predictions
    final_results = pd.concat(results_list, ignore_index=True)

    # Debugging: Print out details about the DataFrame
    print("\nDebugging Information:")
    print(f"Original DataFrame shape: {df.shape}")
    print(f"Final results shape: {final_results.shape}")
    print(f"Number of unique questions: {final_results['question'].nunique()}")
    print(f"Prediction numbers: {final_results['prediction_number'].unique()}")

    return final_results


def main():
    csv_path = 'ragas_20_questions_dataset.csv'


    results = evaluate_all_predictions(csv_path)


    print("\nEvaluation Summary:")
    print(f"Total Questions: {results['question'].nunique()}")

    # Group by question
    grouped_results = results.groupby('question')
    print("\nDetailed Results:")
    for question, group in grouped_results:
        print(f"\nQuestion: {question}")
        # Display predictions for this question
        for index, row in group.iterrows():
            print(f"  Prediction Number: {row['prediction_number']}")
            print(f"  Prediction: {row['prediction']}")
            print(f"  Answer Correctness: {row['answer_correctness']}")



if __name__ == "__main__":
    main()