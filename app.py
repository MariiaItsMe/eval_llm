import ollama
import asyncio
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall
from ragas.metrics import AgentGoalAccuracyWithReference
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class Generation:
    text: str


@dataclass
class LLMResult:
    generations: List[List[Generation]]


class OllamaLLM:
    def __init__(self, host: str = "http://atlas1api.eurecom.fr:8019", model: str = "llama3.1"):
        self.client = ollama.Client(host=host)
        self.model = model

    async def generate(self, prompt: Any, n: Optional[int] = 1, **kwargs) -> LLMResult:
        """
        Generate response using Ollama

        Args:
            prompt: Input text in any format
            n: Number of completions to generate
            **kwargs: Additional arguments (ignored for Ollama compatibility)

        Returns:
            LLMResult with the required format for Ragas
        """
        # Handle any prompt object by extracting text
        if hasattr(prompt, 'text'):
            prompt_text = prompt.text
        else:
            prompt_text = str(prompt)

        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt_text}]
            )
            # Create a list of Generation objects
            generations = [Generation(text=response["message"]["content"])]
            # Wrap in an additional list as per Ragas's expectation
            return LLMResult(generations=[generations])
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return LLMResult(generations=[[Generation(text="")]])

    async def agenerate(self, prompt: Any, n: Optional[int] = 1, **kwargs) -> LLMResult:
        """Async version of generate"""
        return await self.generate(prompt, n, **kwargs)


def create_conversation_sample(question: str, model_response: str, reference: str) -> MultiTurnSample:
    """
    Create a simple conversation sample for evaluation

    Args:
        question: The input question
        model_response: The model's response
        reference: The reference answer for evaluation

    Returns:
        MultiTurnSample: Formatted conversation sample
    """
    return MultiTurnSample(
        user_input=[
            HumanMessage(content=question),
            AIMessage(content=model_response)
        ],
        reference=reference
    )


async def evaluate_conversation(conversation_sample: MultiTurnSample, llm: OllamaLLM) -> float:
    """
    Evaluate a conversation using Ragas metrics

    Args:
        conversation_sample: MultiTurnSample containing the conversation
        llm: OllamaLLM instance

    Returns:
        float: Score from the evaluation
    """
    scorer = AgentGoalAccuracyWithReference()
    scorer.llm = llm

    try:
        score = await scorer.multi_turn_ascore(conversation_sample)
        return score
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return 0.0


async def main():
    # Initialize the Ollama LLM
    llm = OllamaLLM()

    # Example question and reference answer
    question = "Why is the sky blue?"
    reference = "The sky appears blue due to Rayleigh scattering of sunlight in the atmosphere. Shorter wavelengths (blue) are scattered more than longer wavelengths."

    try:
        # Get response from Ollama
        response = llm.client.chat(
            model="llama3.1",
            messages=[{"role": "user", "content": question}]
        )
        model_response = response["message"]["content"]

        print(f"\nQuestion: {question}")
        print(f"Model Response: {model_response}")
        print(f"Reference Answer: {reference}\n")

        # Create the conversation sample
        sample = create_conversation_sample(question, model_response, reference)

        # Evaluate the conversation
        score = await evaluate_conversation(sample, llm)
        print(f"Evaluation score: {score:.5f}")

    except Exception as e:
        print(f"Error during execution: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())