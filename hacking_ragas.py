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
from typing import Optional, Dict
from llama_index.llms.ollama import Ollama
from ragas.llms import LangchainLLMWrapper


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
    statements: t.Optional[SentencesSimplified] = None

    async def _create_simplified_statements(self, question: str, text: str,
                                            callbacks: Callbacks) -> SentencesSimplified:
        result = await super()._create_simplified_statements(question, text, callbacks)
        self.statements = result
        return result

    def init(self, run_config: RunConfig):
        super().init(run_config)


class myRunConfig:
    timeout: Optional[float] = None
    # Add other configuration options as needed


def llm_factory(
        model: str = "llama2",
        run_config: Optional[myRunConfig] = None,
        default_headers: Optional[Dict[str, str]] = None,
        base_url: Optional[str] = None,
) -> BaseRagasLLM:
    """
    Create and return a BaseRagasLLM instance configured for Ollama.

    Parameters
    ----------
    model : str, optional
        The name of the Ollama model to use, by default "llama2"
    run_config : RunConfig, optional
        Configuration for the run, by default None
    default_headers : dict of str, optional
        Default headers to be used in API requests, by default None
    base_url : str, optional
        Base URL for the Ollama API, by default None

    Returns
    -------
    BaseRagasLLM
        An instance of BaseRagasLLM configured with the specified parameters.
    """
    # Extract timeout from run_config if provided
    timeout = None
    if run_config is not None:
        timeout = run_config.timeout

    # Configure Ollama model
    ollama_kwargs = {
        "model": model,
        "timeout": timeout if timeout is not None else 120,  # Default timeout
    }

    # Add base_url if provided
    if base_url is not None:
        ollama_kwargs["base_url"] = base_url

    ollama_model = Ollama(**ollama_kwargs)

    return LangchainLLMWrapper(ollama_model, run_config)

async def evaluate_all_predictions(csv_path):
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

    # Async helper function to process a single data point
    async def process_data_point(data):
        run_config = RunConfig()
        answer_correctness = MyAnswerCorrectness()
        answer_correctness.llm = llm_factory(run_config=run_config) #TODO: we need to push down the LLM and make it Ollama and Mistral-Nemo (3 "ragas")
        answer_correctness.embeddings = embedding_factory(run_config=run_config) #TODO: if the ragas is still better than our own way of splitting and evaluating - we should think about the ways to improve and write the report
        answer_correctness.init(run_config)
        answer_correctness.correctness_prompt = MyCorrectnessPromptWrapper(answer_correctness.correctness_prompt)

        await answer_correctness.single_turn_ascore(data)

        # Collect statements
        statements = getattr(answer_correctness, 'statements', [])

        # Collect evaluations
        evaluations = getattr(answer_correctness.correctness_prompt, 'evaluated_segments', [])

        return statements, evaluations

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

        # Run async operations to process all data points
        results = await asyncio.gather(*[process_data_point(d) for d in dataset])

        # Separate statements and evaluations
        all_statements, all_evaluations = zip(*results)

        # Evaluate the full dataset
        score = evaluate(dataset, metrics=[MyAnswerCorrectness()])

        # Convert to pandas
        results_df = score.to_pandas()

        # Add additional columns
        results_df['prediction_number'] = pred_idx
        results_df['question'] = questions
        results_df['ground_truth'] = ground_truths
        results_df['prediction'] = prediction

        results_df['statements'] = list(all_statements)
        results_df['evaluations'] = list(all_evaluations)

        results_list.append(results_df)

    # Concatenate results from all predictions
    final_results = pd.concat(results_list, ignore_index=True)

    # Debugging: Print out details about the DataFrame
    print("\nDebugging Information:")
    print(f"Number of unique questions: {final_results['question'].nunique()}")
    print(f"Prediction numbers: {final_results['prediction_number'].unique()}")

    # Set pandas options to display full text for debugging
    pd.set_option('display.max_colwidth', None)

    # Print statements and evaluations
    print("\nStatements:")
    print(final_results['statements'].to_string(index=False))
    print("\nEvaluations:")
    for evaluation in final_results['evaluations']:
        print(evaluation)

    # Reset pandas options to default after printing
    pd.reset_option('display.max_colwidth')

    return final_results


async def main():
    csv_path = 'ragas_20_questions_dataset.csv'
    results = await evaluate_all_predictions(csv_path)

    grouped_results = results.groupby('question')
    print("\nDetailed Results:")
    for question, group in grouped_results:
        print(f"\nQuestion: {question}")
        # Display predictions for this question
        for index, row in group.iterrows():
            print(f"  Prediction Number: {row['prediction_number']}")
            print(f"  Prediction: {row['prediction']}")
            print(f"  Answer Correctness: {row['answer_correctness']}")

    return results


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())