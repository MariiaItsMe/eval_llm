import typing as t
import asyncio
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
from typing import List, Optional
import copy
import sys

# Logging wrapper to capture and save terminal output
class LoggingContext:
    def __init__(self, logfile: str):
        self.logfile = logfile

    def __enter__(self):
        self.log_file = open(self.logfile, 'w')
        self.terminal = sys.stdout
        sys.stdout = self
        return self

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.terminal
        self.log_file.close()

class MyCorrectnessPromptWrapper(PydanticPrompt):
    evaluated_segments: Optional[ClassificationWithReason] = None

    def __init__(self, another: PydanticPrompt):
        super().__init__()
        self._another = copy.deepcopy(another)  # Deep copy to prevent state sharing
        self.evaluated_segments = None

    async def generate(self, llm: BaseRagasLLM, data: InputModel, temperature: Optional[float] = None,
                      stop: Optional[List[str]] = None, callbacks: Optional[Callbacks] = None,
                      retries_left: int = 3) -> OutputModel:
        self.evaluated_segments = None
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
        return self._another.process_input(input)

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
    def __init__(self):
        super().__init__()
        self.statements = None
        self._last_processed_text = None

    async def _create_simplified_statements(self, question: str, text: str,
                                            callbacks: Callbacks) -> SentencesSimplified:
        """Override the statement creation to ensure unique processing for each text"""
        try:
            result = await super()._create_simplified_statements(question, text, callbacks)

            # Print debug information
            print("\nDebug Information:")
            print(f"Question: {question}")
            print(f"Text being processed: {text}")
            if result and hasattr(result, 'sentences'):
                print("Generated statements:")
                for sent in result.sentences:
                    print(f"- {sent}")

            self.statements = result
            return result

        except Exception as e:
            print(f"Error in statement generation: {e}")
            print(f"Question: {question}")
            print(f"Text: {text}")
            raise

    async def single_turn_ascore(self, data):
        """Override to ensure fresh processing"""
        self.statements = None
        return await super().single_turn_ascore(data)


async def process_single_prediction(question: str, ground_truth: str, prediction: str,
                                    pred_idx: int, question_idx: int):
    print(f"\nProcessing Prediction {pred_idx} for Question {question_idx}")
    print(f"Prediction text: {prediction}")

    data_samples = {
        'question': [question],
        'ground_truth': [ground_truth],
        'response': [prediction]
    }
    single_dataset = Dataset.from_dict(data_samples)
    single_dataset = convert_v1_to_v2_dataset(single_dataset)
    single_dataset = EvaluationDataset.from_list(single_dataset.to_list())

    run_config = RunConfig()
    answer_correctness = MyAnswerCorrectness()
    answer_correctness.llm = llm_factory(run_config=run_config)
    answer_correctness.embeddings = embedding_factory(run_config=run_config)
    answer_correctness.init(run_config)

    data_point = single_dataset[0]
    await answer_correctness.single_turn_ascore(data_point)
    statements = getattr(answer_correctness, 'statements', None)

    score = evaluate(single_dataset, metrics=[answer_correctness])
    results_df = score.to_pandas()

    results_df['prediction_number'] = pred_idx
    results_df['question'] = question
    results_df['ground_truth'] = ground_truth
    results_df['prediction'] = prediction
    results_df['statements'] = [statements]

    return results_df, statements


async def evaluate_all_predictions(csv_path):
    df = pd.read_csv(csv_path)
    questions = df['question'].tolist()
    ground_truths = df['ground_truth'].tolist()
    predictions = [df[f'prediction_{i}'].tolist() for i in range(1, 5)]

    results_list = []
    output_log = []  # In-memory log

    for pred_idx, prediction_set in enumerate(predictions, 1):
        print(f"\nProcessing prediction set {pred_idx}")
        output_log.append(f"\nProcessing prediction set {pred_idx}\n")

        results = []
        for i, (q, gt, pred) in enumerate(zip(questions, ground_truths, prediction_set)):
            result = await process_single_prediction(q, gt, pred, pred_idx, i)
            results.append(result)

        results_dfs, all_statements = zip(*results)

        for i, (q, p, stmts) in enumerate(zip(questions, prediction_set, all_statements)):
            log_entry = [
                f"\n{'=' * 50}",
                f"Index: {i}, Prediction {pred_idx}",
                f"Question: {q}",
                f"Prediction: {p}",
                "Generated Statements:"
            ]

            if stmts and hasattr(stmts, 'sentences'):
                for sent in stmts.sentences:
                    if isinstance(sent, dict) and 'simpler_statements' in sent:
                        for stmt in sent['simpler_statements']:
                            log_entry.append(f"- {stmt}")
                    else:
                        log_entry.append(f"- {sent}")
            else:
                log_entry.append("No statements generated")

            log_entry.append("")  # Add a blank line
            output_log.extend(log_entry)

        results_list.extend(results_dfs)

    # Print all logs at the end or process them as needed
    for entry in output_log:
        print(entry)

    final_results = pd.concat(results_list, ignore_index=True)
    return final_results


async def main():
    csv_path = 'ragas_20_questions_dataset.csv'
    results = await evaluate_all_predictions(csv_path)
    return results


if __name__ == "__main__":
    load_dotenv()
    log_file = "terminal_output.log"
    with LoggingContext(log_file):
        asyncio.run(main())
