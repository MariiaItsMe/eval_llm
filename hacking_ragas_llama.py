import os
import logging
import pandas as pd
from dotenv import load_dotenv
from ragas.metrics import AnswerCorrectness
from ragas import evaluate, EvaluationDataset
from llama_index.llms.ollama import Ollama
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_evaluator():
    """Initialize and return the LLM and embeddings for evaluation."""
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables or .env file")

        # Initialize LLM
        base_url = os.getenv('BASE_URL', 'http://atlas1api.eurecom.fr:8019')
        llm = Ollama(model="llama3.1:70b", base_url=base_url)
        evaluator_llm = LlamaIndexLLMWrapper(llm)

        # Initialize Embeddings
        openai_embedding = OpenAIEmbedding(
            model="text-embedding-ada-002",
            api_key=api_key
        )
        embeddings = LlamaIndexEmbeddingsWrapper(embeddings=openai_embedding)

        return evaluator_llm, embeddings

    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {str(e)}")
        raise


def evaluate_predictions(csv_path: str) -> pd.DataFrame:
    """
    Evaluate answer correctness for all predictions.
    """
    try:
        # Initialize components
        evaluator_llm, embeddings = initialize_evaluator()

        # Load and validate data
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        required_columns = ['question', 'ground_truth']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")

        # Remove any rows with null values
        df = df.dropna(subset=['question', 'ground_truth'])

        # Get prediction columns
        prediction_columns = [col for col in df.columns if col.startswith('prediction_')]
        if not prediction_columns:
            raise ValueError("No prediction columns found in CSV")

        # Initialize metric
        answer_correctness = AnswerCorrectness(llm=evaluator_llm, embeddings=embeddings)
        all_results = []

        # Process each question for each prediction column
        for pred_idx, prediction_col in enumerate(prediction_columns, 1):
            logger.info(f"Processing predictions from column: {prediction_col}")

            # Process each question individually
            for idx, row in df.iterrows():
                eval_data = [{
                    'user_input': row['question'],
                    'response': row[prediction_col],
                    'reference': row['ground_truth']
                }]

                # Create dataset and evaluate
                dataset = EvaluationDataset.from_list(eval_data)
                try:
                    results = evaluate(dataset, metrics=[answer_correctness])
                    results_df = results.to_pandas()

                    # Add metadata
                    results_df['prediction_number'] = pred_idx
                    results_df['question'] = row['question']
                    results_df['ground_truth'] = row['ground_truth']
                    results_df['prediction'] = row[prediction_col]

                    all_results.append(results_df)
                    logger.info(f"Evaluated question {idx + 1}/{len(df)} for prediction {pred_idx}")
                except Exception as e:
                    logger.error(f"Failed to evaluate question {idx + 1} for prediction {pred_idx}: {str(e)}")

        # Combine results
        final_results = pd.concat(all_results, ignore_index=True)

        # Verify all predictions were evaluated
        expected_count = len(df) * len(prediction_columns)
        if len(final_results) != expected_count:
            logger.warning(f"Expected {expected_count} evaluations but got {len(final_results)}")
            missing_mask = final_results['answer_correctness'].isna()
            if missing_mask.any():
                logger.warning(f"Found {missing_mask.sum()} missing evaluations")

        return final_results

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


def main():
    try:
        csv_path = 'ragas_20_questions_dataset.csv'
        results = evaluate_predictions(csv_path)

        # Print summary statistics
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

        # Check for missing evaluations
        missing_evals = results['answer_correctness'].isna().sum()
        if missing_evals > 0:
            print(f"\nWarning: {missing_evals} evaluations are missing or failed")

    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()