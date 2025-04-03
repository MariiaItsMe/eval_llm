import asyncio
import json
from typing import List, Optional, Type

import numpy as np
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
from ragas.metrics._faithfulness import NLIStatementPrompt
from ragas.metrics.utils import fbeta_score
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from models import AssessmentDataset, StatementEval, FactualCorrectnessEval, StatementPairEval, \
    AltFactualCorrectnessEval, AssessmentEntry


class AnnotatedDataset(EvaluationDataset):
    def get_sample_type(self) -> Type[SingleTurnSample]:
        return SingleTurnSample  # Force it to return the base class


class AnnotatedSingleTurnSample(SingleTurnSample):
    ref_score: Optional[str] = None
    reference_claims: Optional[List[str]] = None
    response_claims: Optional[List[str]] = None


class HackedFactualCorrectness(FactualCorrectness):

    def __post_init__(self):
        super().__post_init__()
        self.nli_prompt = NLIStatementPrompt()

    def _only_required_columns_single_turn(self, sample: SingleTurnSample) -> SingleTurnSample:
        return sample

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks: Callbacks) -> float:
        assert isinstance(sample, AnnotatedSingleTurnSample)
        reference = sample.reference
        response = sample.response
        reference_claims = sample.reference_claims
        response_claims = sample.response_claims

        assert self.llm is not None, "LLM must be set"
        assert reference is not None, "Reference is not set"
        assert response is not None, "Response is not set"
        assert response_claims is not None, "Response claims is not set"

        split_rm, recall_cb = new_group(
            name="claim_decomposition_prompt",
            inputs={"data": response},
            callbacks=callbacks,
            metadata={"type": "nli-evaluation"},
        )
        split_rm.on_chain_end({"output": {"claims": response_claims}})

        # response_claims = await self.decompose_claims(response, callbacks)
        self.nli_prompt.name = "precision_eval"
        reference_response = await self.verify_claims(
            premise=reference, hypothesis_list=response_claims, callbacks=callbacks
        )

        if self.mode != "precision":
            # reference_claims = await self.decompose_claims(reference, callbacks)
            self.nli_prompt.name = "recall_eval"
            response_reference = await self.verify_claims(
                premise=response, hypothesis_list=reference_claims, callbacks=callbacks
            )
        else:
            response_reference = np.array([], dtype=bool)

        tp = sum(reference_response)
        fp = sum(~reference_response)
        if self.mode != "precision":
            fn = sum(~response_reference)
        else:
            fn = 0

        if self.mode == "precision":
            score = tp / (tp + fp + 1e-8)
        elif self.mode == "recall":
            score = tp / (tp + fn + 1e-8)
        else:
            score = fbeta_score(tp, fp, fn, self.beta)

        return np.round(score, 2)


class AltFactualCorrectness(FactualCorrectness):
    # _required_columns: Dict[MetricType, Set[str]] = field(
    #     default_factory=lambda: {
    #         MetricType.SINGLE_TURN: {"response", "reference", "reference_claims", "response_claims"}
    #     }
    # )

    label_mapping = ['contradiction', 'entailment', 'neutral']
    nli_model: Optional[CrossEncoder] = None

    def _only_required_columns_single_turn(self, sample: SingleTurnSample) -> SingleTurnSample:
        return sample

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
        assert isinstance(sample, AnnotatedSingleTurnSample)
        reference = sample.reference
        response = sample.response
        reference_claims = sample.reference_claims
        response_claims = sample.response_claims

        assert self.nli_model is not None, "CrossEncoder must be loaded"
        assert self.llm is not None, "LLM must be set"
        assert reference is not None, "Reference is not set"
        assert response is not None, "Response is not set"
        assert reference_claims is not None, "Reference claims is not set"
        assert response_claims is not None, "Response claims is not set"

        split_rm, recall_cb = new_group(
            name="reference_claim_decomposition_prompt",
            inputs={"data": reference},
            callbacks=callbacks,
            metadata={"type": "nli-evaluation"},
        )
        recall_cb.on_chain_end({"output": {"claims": reference_claims}})

        split_rm, recall_cb = new_group(
            name="response_claim_decomposition_prompt",
            inputs={"data": response},
            callbacks=callbacks,
            metadata={"type": "nli-evaluation"},
        )
        recall_cb.on_chain_end({"output": {"claims": response_claims}})

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
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    new_fc = AltFactualCorrectness(llm=evaluator_llm, atomicity="high", coverage="high")
    fc = HackedFactualCorrectness(llm=evaluator_llm, atomicity="high", coverage="high")
    # question = "How does COVID-19 spreads?"
    # expert = "COVID-19 spreads via airborne droplets. Vaccines reduce severe outcomes. Most severe outcomes involves difficulty for breathing."
    # llm = "The COVID-19 virus transmits through air. To control COVID-19, governments imposed lockdowns. Vaccines reduced hospitalization risk through a reduction of severe cases."

    with open("dataset.json", "r") as file:
        data = json.load(file)

    # data = data[:1]

    eval_samples = []
    for i in tqdm(data, desc="Processing: "):
        reference_claims = asyncio.run(fc.decompose_claims(i["ground_truth"], []))
        for ref_score, response in i["answers"].items():
            response_claims = asyncio.run(fc.decompose_claims(response, []))

            eval_samples.append(
                AnnotatedSingleTurnSample(
                    user_input=i["question"],
                    response=response,
                    reference=i["ground_truth"],
                    ref_score=ref_score,
                    reference_claims=reference_claims,
                    response_claims=response_claims
                ))

    eval_dataset = AnnotatedDataset(samples=eval_samples)

    results = evaluate(eval_dataset, metrics=[new_fc, fc])

    assessment_dataset = AssessmentDataset(llm_model="gpt-4o-mini", nli_model="cross-encoder/nli-deberta-v3-large")

    for trace, sample in zip(results.traces, results.dataset.samples):
        question = sample.user_input
        reference = sample.reference
        answer = sample.response
        ref_score = sample.ref_score

        # factual correctness
        score = float(trace.scores["factual_correctness"])
        precision_statements = [
            StatementEval(statement=s.statement, reason=s.reason, verdict=s.verdict)
            for s in trace["factual_correctness"]["precision_eval"]["output"].statements
        ]

        recall_statements = [
            StatementEval(statement=s.statement, reason=s.reason, verdict=s.verdict)
            for s in trace["factual_correctness"]["recall_eval"]["output"].statements
        ]

        claims = trace["factual_correctness"]["claim_decomposition_prompt"]["output"]["claims"]

        factual_correctness_eval = FactualCorrectnessEval(
            score=score, claims=claims, precision_statements=precision_statements, recall_statements=recall_statements
        )

        # alt factual correctness
        alt_score = float(trace.scores["alt_factual_correctness"])
        reference_claims = trace["alt_factual_correctness"]["reference_claim_decomposition_prompt"]["output"]["claims"]
        answer_claims = trace["alt_factual_correctness"]["response_claim_decomposition_prompt"]["output"]["claims"]

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

    with open("./assessment-2.json", "w", encoding="utf-8") as f:
        f.write(assessment_dataset.model_dump_json(indent=4))
