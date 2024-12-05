import time
import typing as t

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_core.callbacks import Callbacks
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.embeddings import embedding_factory
from ragas.llms import BaseRagasLLM, llm_factory
from ragas.metrics import AnswerCorrectness
from ragas.metrics._answer_correctness import ClassificationWithReason
from ragas.metrics._faithfulness import SentencesSimplified
from ragas.prompt import PydanticPrompt, InputModel, OutputModel
from ragas.utils import convert_v1_to_v2_dataset


class MyCorrectnessPromptWrapper(PydanticPrompt):
    evaluated_segments: t.Optional[ClassificationWithReason] = None

    def __init__(self, another: PydanticPrompt):
        super().__init__()
        self._another = another

    async def generate(self, llm: BaseRagasLLM, data: InputModel, temperature: t.Optional[float] = None,
                       stop: t.Optional[t.List[str]] = None, callbacks: t.Optional[Callbacks] = None,
                       retries_left: int = 3) -> OutputModel:
        result = await self._another.generate(llm, data, temperature, stop, callbacks, retries_left)
        self.evaluated_segments = result
        return result

    def to_string(self, data: t.Optional[InputModel] = None) -> str:
        return self._another.to_string(data)

    async def generate_multiple(self, llm: BaseRagasLLM, data: InputModel, n: int = 1,
                                temperature: t.Optional[float] = None, stop: t.Optional[t.List[str]] = None,
                                callbacks: t.Optional[Callbacks] = None, retries_left: int = 3) -> t.List[OutputModel]:
        return await super().generate_multiple(llm, data, n, temperature, stop, callbacks, retries_left)

    def process_input(self, input: InputModel) -> InputModel:
        return self._another.to_string(input)

    def process_output(self, output: OutputModel, input: InputModel) -> OutputModel:
        return self._another.process_output(output, input)

    async def adapt(self, target_language: str, llm: BaseRagasLLM,
                    adapt_instruction: bool = False) -> "PydanticPrompt[InputModel, OutputModel]":
        return await self._another.adapt(target_language, llm, adapt_instruction)

    def save(self, file_path: str):
        return self._another.save(file_path)

    @classmethod
    def load(cls, file_path: str) -> "PydanticPrompt[InputModel, OutputModel]":
        raise NotImplementedError

    def __repr__(self):
        return (f"{self.__class__.__name__}(instruction={self._another.instruction}, "
                f"examples={self._another.examples}, "
                f"language={self._another.language})")


class MyAnswerCorrectness(AnswerCorrectness):
    statements: t.Optional[SentencesSimplified] = None

    async def _create_simplified_statements(self, question: str, text: str,
                                            callbacks: Callbacks) -> SentencesSimplified:
        result = await super()._create_simplified_statements(question, text, callbacks)
        self.statements = result
        return result

    def init(self, run_config: RunConfig):
        super().init(run_config)


def evaluate_all_predictions(csv_path):
    """
    Evaluate answer correctness using Ragas metrics for all predictions in a given CSV dataset
    """
    df = pd.read_csv(csv_path)
    df = df[:3]
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
        dataset = convert_v1_to_v2_dataset(dataset)
        dataset = EvaluationDataset.from_list(dataset.to_list())

        time.sleep(1)  # 1-second pause between operations
        # Evaluate using Ragas
        run_config = RunConfig()
        for d in dataset:
            answer_correctness = MyAnswerCorrectness()
            answer_correctness.llm = llm_factory(run_config=run_config)  # TODO: replace with Ollama
            answer_correctness.embeddings = embedding_factory(
                run_config=run_config)  # TODO: replace with Ollama embeddings
            answer_correctness.init(run_config)
            answer_correctness.correctness_prompt = MyCorrectnessPromptWrapper(answer_correctness.correctness_prompt)

            answer_correctness.single_turn_score(d)
            statements = answer_correctness.statements  # TODO: you can get the statements here.
            evaluations = answer_correctness.correctness_prompt.evaluated_segments
            pass

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
    load_dotenv()
    main()
