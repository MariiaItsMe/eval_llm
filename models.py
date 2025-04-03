from typing import List

import pydantic


class StatementEval(pydantic.BaseModel):
    statement: str
    reason: str
    verdict: int


class FactualCorrectnessEval(pydantic.BaseModel):
    score: float
    claims: List[str]
    precision_statements: List[StatementEval]
    recall_statements: List[StatementEval]


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
