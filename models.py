from pydantic import BaseModel, Field, field_validator
from typing import Any, Literal, TypedDict

FailureType = Literal[
    "hallucination",
    "refusal",
    "instruction_violation",
    "reasoning_error",
    "tool_misuse",
    "none",
]

ConsensusFlag = Literal["clean", "verified", "weak_consensus", "disagreement", "pass"]


class EvalState(TypedDict):
    seed_id: str
    seed_prompt: str
    expected_behavior: str
    reference_answer: str
    ambiguity_note: str
    category: str
    adversarial_prompt: str
    mutation_operator: str
    mutation_history: list[dict]
    target_response: str
    tool_trace: list[dict]
    judge_results: list[dict]
    target_failed: bool
    consensus_flag: str
    difficulty_score: float
    iteration: int
    done: bool


VALID_FAILURE_TYPES = {
    "hallucination",
    "refusal",
    "instruction_violation",
    "reasoning_error",
    "tool_misuse",
    "none",
}

# judges sometimes return null or junk,clean it up here
def normalize_judge_data(data: dict[str, Any]) -> dict[str, Any]:
    
    normalized = dict(data)
    failure_detected = bool(normalized.get("failure_detected", False))

    failure_type = normalized.get("failure_type")
    if failure_type is None or str(failure_type).strip().lower() in ("", "null", "none"):
        normalized["failure_type"] = "reasoning_error" if failure_detected else "none"
    else:
        cleaned_type = str(failure_type).strip().lower().replace(" ", "_")
        if cleaned_type in VALID_FAILURE_TYPES:
            normalized["failure_type"] = cleaned_type
        else:
            normalized["failure_type"] = "reasoning_error" if failure_detected else "none"

    confidence = normalized.get("confidence")
    try:
        normalized["confidence"] = float(confidence) if confidence is not None else 0.5
    except (TypeError, ValueError):
        normalized["confidence"] = 0.5
    normalized["confidence"] = max(0.0, min(1.0, normalized["confidence"]))

    normalized["reason"] = str(normalized.get("reason") or "").strip() or "no reason given"
    normalized["failure_detected"] = failure_detected
    return normalized


class JudgeVerdict(BaseModel):
    failure_detected: bool
    failure_type: FailureType = "none"
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str

    @field_validator("failure_type", mode="before")
    @classmethod
    def coerce_failure_type(cls, value: Any) -> str:
        if value is None or str(value).strip().lower() in ("", "null"):
            return "none"
        ft = str(value).strip().lower().replace(" ", "_")
        return ft if ft in VALID_FAILURE_TYPES else "none"


class MutationRecord(BaseModel):
    operator: str
    prompt: str
    reward: float


class AttackerOutput(BaseModel):
    adversarial_prompt: str


class AuditResult(BaseModel):
    quality_score: int = Field(ge=1, le=5)
    is_real_failure: bool
    summary: str
