import json
from typing import List, Optional, Type

import numpy as np
import pydantic
import torch
from langchain_core.callbacks import Callbacks
from langchain_openai import ChatOpenAI
from numpy.typing import NDArray
from ragas import evaluate
from ragas.callbacks import new_group
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness
from ragas.metrics._factual_correctness import ClaimDecompositionPrompt
from sentence_transformers import CrossEncoder


class AnnotatedDataset(EvaluationDataset):
    def get_sample_type(self) -> Type[SingleTurnSample]:
        return SingleTurnSample  # Force it to return the base class


class AnnotatedSingleTurnSample(SingleTurnSample):
    ref_score: Optional[str] = None


class StatementEval(pydantic.BaseModel):
    statement: str
    reason: str
    verdict: int


class FactualCorrectnessEval(pydantic.BaseModel):
    score: float
    claims: List[str]
    statement_evals: List[StatementEval]


class StatementPairEval(pydantic.BaseModel):
    premise: str
    hypothesis: str
    logits: List[float]
    classification: str


class AltFactualCorrectnessEval(pydantic.BaseModel):
    score: float
    reference_claims: List[str]
    answer_claims: List[str]
    precision_statements: List[StatementPairEval]
    recall_statements: List[StatementPairEval]


class AssessmentEntry(pydantic.BaseModel):
    question: str
    reference: str
    answer: str
    ref_score: str

    ragas: FactualCorrectnessEval
    ours: AltFactualCorrectnessEval


class AssessmentDataset(pydantic.BaseModel):
    llm_model: str
    nli_model: str
    entries: List[AssessmentEntry] = []


class AltFactualCorrectness(FactualCorrectness):
    label_mapping = ['contradiction', 'entailment', 'neutral']
    nli_model: Optional[CrossEncoder] = None

    def __post_init__(self):
        super().__post_init__()

        model = CrossEncoder(
            "cross-encoder/nli-deberta-v3-large",
            tokenizer_args={"use_fast": False}
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.model.to(device)
        self.nli_model = model
        self.name = "alt_factual_correctness"
        self.claim_decomposition_prompt = ClaimDecompositionPrompt()

    async def verify_claims(
            self, premise: str, hypothesis_list: List[str], callbacks: Callbacks
    ) -> NDArray[np.bool_]:
        raise NotImplementedError

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks: Callbacks) -> float:
        reference = sample.reference
        response = sample.response
        assert self.nli_model is not None, "CrossEncoder must be loaded"
        assert self.llm is not None, "LLM must be set"
        assert reference is not None, "Reference is not set"
        assert response is not None, "Response is not set"

        self.claim_decomposition_prompt.name = "response_claim_decomposition_prompt"
        response_claims = await self.decompose_claims(response, callbacks)

        self.claim_decomposition_prompt.name = "reference_claim_decomposition_prompt"
        reference_claims = await self.decompose_claims(reference, callbacks)

        precision_pairs = [(ref, resp) for resp in response_claims for ref in reference_claims]

        # Compute precision
        precision_rm, precision_cb = new_group(
            name="precision_eval",
            inputs={"data": precision_pairs},
            callbacks=callbacks,
            metadata={"type": "nli-evaluation"},
        )

        precision_scores = self.nli_model.predict(precision_pairs)
        precision_labels = [self.label_mapping[score_max] for score_max in precision_scores.argmax(axis=1)]
        precision_entailments = np.array([label == "entailment" for label in precision_labels], dtype=bool)
        precision_matrix = precision_entailments.reshape(len(response_claims), len(reference_claims))

        precision_rm.on_chain_end({"output": {"classification": precision_labels, "logits": precision_scores}})

        TP_p = np.sum(np.any(precision_matrix, axis=1))
        FP_p = len(response_claims) - TP_p

        # Compute recall
        recall_rm, recall_cb = new_group(
            name="recall_eval",
            inputs={"data": precision_pairs},
            callbacks=callbacks,
            metadata={"type": "nli-evaluation"},
        )

        recall_pairs = [(resp, ref) for resp in response_claims for ref in reference_claims]
        recall_scores = self.nli_model.predict(recall_pairs)
        recall_labels = [self.label_mapping[score_max] for score_max in recall_scores.argmax(axis=1)]
        recall_entailments = np.array([label == "entailment" for label in recall_labels], dtype=bool)
        recall_matrix = recall_entailments.reshape(len(reference_claims), len(response_claims))

        recall_rm.on_chain_end({"output": {"classification": recall_labels, "logits": recall_scores}})

        TP_r = np.sum(np.any(recall_matrix, axis=1))
        FN_r = len(reference_claims) - TP_r

        precision = TP_p / (TP_p + FP_p) if (TP_p + FP_p) > 0 else 0.0
        recall = TP_r / (TP_r + FN_r) if (TP_r + FN_r) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return np.round(f1, 2)


if __name__ == "__main__":
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    new_fc = AltFactualCorrectness(llm=evaluator_llm, atomicity="high", coverage="high")
    fc = FactualCorrectness(llm=evaluator_llm, atomicity="high", coverage="high")
    # question = "How does COVID-19 spreads?"
    # expert = "COVID-19 spreads via airborne droplets. Vaccines reduce severe outcomes. Most severe outcomes involves difficulty for breathing."
    # llm = "The COVID-19 virus transmits through air. To control COVID-19, governments imposed lockdowns. Vaccines reduced hospitalization risk through a reduction of severe cases."

    with open("ai_generated_dataset_llm_judge.json", "r") as file:
        data = json.load(file)

    eval_dataset = AnnotatedDataset(samples=[
        AnnotatedSingleTurnSample(user_input=i["question"],
                                  response=response,
                                  reference=i["ground_truth"],
                                  ref_score=ref_score)
        for i in data for ref_score, response in list(data[0]["answers"].items())
    ])

    results = evaluate(eval_dataset, metrics=[new_fc, fc])

    assessment_dataset = AssessmentDataset(llm_model="gpt-4o-mini", nli_model="cross-encoder/nli-deberta-v3-large")

    for trace, sample in zip(results.traces, results.dataset.samples):
        question = sample.user_input
        reference = sample.reference
        answer = sample.response
        ref_score = sample.ref_score

        # factual correctness
        score = float(trace.scores["factual_correctness"])
        nli_statements = [
            StatementEval(statement=s.statement, reason=s.reason, verdict=s.verdict)
            for s in trace["factual_correctness"]["n_l_i_statement_prompt"]["output"].statements
        ]

        claims = trace["factual_correctness"]["claim_decomposition_prompt"]["output"].claims

        factual_correctness_eval = FactualCorrectnessEval(score=score, claims=claims, statement_evals=nli_statements)

        # alt factual correctness
        alt_score = float(trace.scores["alt_factual_correctness"])
        reference_claims = trace["alt_factual_correctness"]["reference_claim_decomposition_prompt"]["output"].claims
        answer_claims = trace["alt_factual_correctness"]["response_claim_decomposition_prompt"]["output"].claims

        precision_pairs = trace["alt_factual_correctness"]["precision_eval"]["input"]
        precision_classification = trace["alt_factual_correctness"]["precision_eval"]["output"]["classification"]
        precision_logits = trace["alt_factual_correctness"]["precision_eval"]["output"]["logits"]

        recall_pairs = trace["alt_factual_correctness"]["recall_eval"]["input"]
        recall_classification = trace["alt_factual_correctness"]["recall_eval"]["output"]["classification"]
        recall_logits = trace["alt_factual_correctness"]["recall_eval"]["output"]["logits"]

        precision_evals = [
            StatementPairEval(premise=p[0], hypothesis=p[1], logits=l, classification=c)
            for p, c, l in zip(precision_pairs, precision_classification, precision_logits)
        ]

        recall_evals = [
            StatementPairEval(premise=p[0], hypothesis=p[1], logits=l, classification=c)
            for p, c, l in zip(recall_pairs, recall_classification, recall_logits)
        ]

        alt_factual_correctness_eval = AltFactualCorrectnessEval(
            score=alt_score,
            reference_claims=reference_claims,
            answer_claims=answer_claims,
            precision_statements=precision_evals,
            recall_statements=recall_evals
        )

        assessment_dataset.entries.append(
            AssessmentEntry(
                question=question,
                reference=reference,
                answer=answer,
                ref_score=ref_score,
                ragas=factual_correctness_eval,
                ours=alt_factual_correctness_eval
            )
        )

    with open("./assessment.json", "w", encoding="utf-8") as f:
        f.write(assessment_dataset.model_dump_json(indent=4))
