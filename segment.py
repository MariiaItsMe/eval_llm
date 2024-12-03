import typing as t
import time
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas.metrics import answer_correctness
from ragas import evaluate


from langchain_core.callbacks import Callbacks
from ragas.llms import BaseRagasLLM
from ragas.metrics import AnswerCorrectness
from ragas.metrics._faithfulness import SentencesSimplified
from ragas.prompt import PydanticPrompt, InputModel, OutputModel


# Load environment variables
load_dotenv()

class MyCorrectnessPrompt(PydanticPrompt):

    async def generate(self, llm: BaseRagasLLM, data: InputModel, temperature: t.Optional[float] = None,
                       stop: t.Optional[t.List[str]] = None, callbacks: t.Optional[Callbacks] = None,
                       retries_left: int = 3) -> OutputModel:
        result = await super().generate(llm, data, temperature, stop, callbacks, retries_left)
        callbacks.metadata["evaluated_segments"] = result
        return result


class MyAnswerCorrectness(AnswerCorrectness):

    async def _create_simplified_statements(self, question: str, text: str,
                                            callbacks: Callbacks) -> SentencesSimplified:
        result = await super()._create_simplified_statements(question, text, callbacks)
        callbacks.metadata["statements"] = result
        return result


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
        score = evaluate(dataset, metrics=[MyAnswerCorrectness()])

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
    result = evaluate_all_predictions(csv_path)


if __name__ == "__main__":
    main()