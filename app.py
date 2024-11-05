import ollama
import pandas as pd
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithReference
import asyncio

async def evaluate_ollama_performance():
    # Load your Excel dataset
    data = pd.read_json("/home/guendouz/sources/eval_llm/dataset.json")  # Adjust the path as needed
    # Initialize the Ragas context precision metric
    context_precision = LLMContextPrecisionWithReference()
    # Initialize the Ollama client
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
    # Prepare to store scores
    scores = []
    # Iterate through each question-answer pair
    for index, row in data.iterrows():
        # Get the question and correct answer
        question = row['question']
        correct_answer = row['reference_answer']
        # Communicate with Ollama to get the response
        response = client.chat(model="llama3.1", messages=[{"role": "user", "content": question}])  # Ensure this call is also awaited
        ollama_response = response["message"]["content"]
        # Create a SingleTurnSample for evaluation
        sample = SingleTurnSample(
            user_input=question,
            reference=correct_answer,
            retrieved_contexts=[ollama_response]  # Use the response from Ollama
        )
        # Calculate context precision score
        score = context_precision.single_turn_ascore(sample)
        scores.append(score)
    # Add the scores to the DataFrame
    data['context_precision_score'] = scores
    # Print or return the results as needed
    print(data[['question', 'reference_answer', 'context_precision_score']])

# Entry point of the script
if __name__ == "__main__":
    asyncio.run(evaluate_ollama_performance())
