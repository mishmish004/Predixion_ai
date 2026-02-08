from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CallType(str, Enum):
    REMINDER = "reminder"
    FOLLOW_UP = "follow_up"
    ESCALATION = "escalation"
    SETTLEMENT = "settlement"


class CallOutcome(str, Enum):
    PTP = "ptp"
    PARTIAL = "partial"
    CALLBACK = "callback"
    REFUSED = "refused"
    ESCALATE = "escalate"


class MessageTurn(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None


class ConversationMetadata(BaseModel):
    outstanding_amount: float
    days_past_due: int
    product_type: str
    customer_segment: str


class ConversationInput(BaseModel):
    conversation_id: str
    call_type: str
    messages: list[MessageTurn]
    metadata: ConversationMetadata


class CriticalEvent(BaseModel):
    event_type: str = Field(
        description="Type: THREAT, EMPATHY, HARDSHIP_CLAIM, PTP, NEGOTIATION, COMPLIANCE_VIOLATION, GREETING, CLOSING"
    )
    speaker: str = Field(description="agent or customer")
    description: str
    severity: str = Field(default="low", description="low, medium, high")


class IntentTurn(BaseModel):
    turn_number: int
    speaker: str
    intent: str
    content_summary: str


class StateMap(BaseModel):
    conversation_id: str
    call_type: str
    intent_flow: list[IntentTurn] = Field(default_factory=list)
    critical_events: list[CriticalEvent] = Field(default_factory=list)
    customer_sentiment_trajectory: str = Field(default="neutral")
    agent_compliance_summary: str = Field(default="")
    negotiation_outcome: str = Field(default="")
    hardship_indicators: list[str] = Field(default_factory=list)
    overall_tone: str = Field(default="neutral")


class GroundTruthScores(BaseModel):
    overall_score: int = Field(ge=1, le=5)
    compliance_ethics: int = Field(ge=1, le=5)
    goal_achievement: int = Field(ge=1, le=5)
    communication_quality: int = Field(ge=1, le=5)
    empathy_tone: int = Field(ge=1, le=5)
    objection_handling: int = Field(ge=1, le=5)


class GroundTruth(BaseModel):
    conversation_id: str
    call_type: str
    metadata: ConversationMetadata
    scores: GroundTruthScores
    outcome: str
    compliance_flags: list[str] = Field(default_factory=list)
    notes: str = ""
    state_map_text: str = Field(
        default="",
        description="Serialized summary of the state map used for vector embedding",
    )


class PrecedentMatch(BaseModel):
    ground_truth: GroundTruth
    similarity_score: float


class AuditDimensionScore(BaseModel):
    score: int = Field(ge=1, le=5)
    explanation: str


class AuditResult(BaseModel):
    conversation_id: str
    overall_score: int = Field(ge=1, le=5)
    dimensions: dict[str, AuditDimensionScore]
    outcome: str
    summary: str
    compliance_flags: list[str] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)
    sentiment_shifts: list[str] = Field(default_factory=list)
    precedent_conversation_id: Optional[str] = None
    precedent_similarity_score: Optional[float] = None
