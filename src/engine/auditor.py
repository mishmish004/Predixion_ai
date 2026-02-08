from typing import Optional

import orjson

from src.core.config import settings
from src.core.schema import (
    AuditDimensionScore,
    AuditResult,
    ConversationInput,
    PrecedentMatch,
    StateMap,
)
from src.infra.clients import LLMClient


AUDIT_SYSTEM_PROMPT = """You are a senior quality auditor for Indian debt collection calls. You audit calls against RBI (Reserve Bank of India) Fair Practices Code and industry best practices.

You will receive:
1. The original Hinglish conversation transcript
2. A structured "State Map" extracted from the conversation
3. A "Historical Precedent" - a similar past call that was already graded by human auditors (this may be absent if no close match was found)

Your job is to produce a final audit score. If a Precedent is provided, use it as a calibration anchor: compare the current call's behavior against the precedent to ensure consistent grading. If no Precedent is available, grade purely on the rubric below.

SCORING RUBRIC (1-5 scale for each dimension):

COMPLIANCE & ETHICS:
5 = Perfect compliance. Proper identity disclosure, no threats, respectful, followed all RBI guidelines.
4 = Minor gaps (e.g., slightly incomplete introduction) but no violations.
3 = Some concerning behavior but no clear violations.
2 = One clear compliance issue (e.g., implied threat, dismissive of hardship).
1 = Severe violations - explicit threats, harassment, abusive language, calling at wrong hours.

GOAL ACHIEVEMENT:
5 = Secured full payment commitment with clear date/amount, or best possible outcome for the situation.
4 = Secured partial payment or callback with specific timeline.
3 = Some progress made but commitment is vague.
2 = Minimal progress, customer non-committal.
1 = No progress, call ended badly, customer refused or hung up.

COMMUNICATION QUALITY:
5 = Clear, professional, natural Hinglish. Explained all details (amount, due date, consequences, options). Active listening.
4 = Good communication with minor gaps in information sharing.
3 = Adequate but could be clearer. Some information missing.
2 = Poor - minimal responses, made customer work hard for basic information.
1 = Very poor - unclear, confusing, or inappropriate communication.

EMPATHY & TONE:
5 = Genuinely empathetic, acknowledged customer's situation, professional yet caring. Adapted tone appropriately.
4 = Showed empathy but could have been warmer or more adaptive.
3 = Neutral tone, neither empathetic nor cold.
2 = Cold or dismissive, minimal acknowledgment of customer's situation.
1 = Hostile, rude, or completely insensitive to customer's hardship.

OBJECTION HANDLING:
5 = Expertly handled all objections. Offered relevant solutions (EMI plans, settlements, extensions). Addressed every concern.
4 = Handled objections well with some solutions offered.
3 = Addressed some objections but missed opportunities for solutions.
2 = Poor handling - dismissed objections without alternatives.
1 = Failed completely - ignored or worsened objections.

OUTCOME CLASSIFICATION:
- ptp: Customer made a promise to pay (full or partial) with specific date
- partial: Partial payment was made or committed
- callback: Follow-up call was scheduled
- refused: Customer refused to pay
- escalate: Needs supervisor or legal review

You MUST respond with ONLY valid JSON (no markdown, no code blocks):
{
  "conversation_id": "string",
  "overall_score": integer 1-5,
  "dimensions": {
    "compliance_ethics": {"score": integer 1-5, "explanation": "string"},
    "goal_achievement": {"score": integer 1-5, "explanation": "string"},
    "communication_quality": {"score": integer 1-5, "explanation": "string"},
    "empathy_tone": {"score": integer 1-5, "explanation": "string"},
    "objection_handling": {"score": integer 1-5, "explanation": "string"}
  },
  "outcome": "ptp|partial|callback|refused|escalate",
  "summary": "2-3 sentence overall assessment",
  "compliance_flags": ["list of specific compliance issues found, empty if none"],
  "improvement_suggestions": ["list of actionable coaching tips for the agent"],
  "sentiment_shifts": ["list describing customer sentiment changes during the call"]
}"""


def build_audit_user_prompt(
    conversation: ConversationInput,
    state_map: StateMap,
    precedent_match: Optional[PrecedentMatch],
) -> str:
    transcript_lines = []
    for index, message in enumerate(conversation.messages):
        speaker_label = message.role.upper()
        transcript_lines.append(f"Turn {index + 1} [{speaker_label}]: {message.content}")
    transcript_text = "\n".join(transcript_lines)

    state_map_summary = (
        f"Call Type: {state_map.call_type}\n"
        f"Overall Tone: {state_map.overall_tone}\n"
        f"Customer Sentiment: {state_map.customer_sentiment_trajectory}\n"
        f"Compliance Summary: {state_map.agent_compliance_summary}\n"
        f"Negotiation Outcome: {state_map.negotiation_outcome}\n"
        f"Hardship Indicators: {', '.join(state_map.hardship_indicators) if state_map.hardship_indicators else 'None'}\n"
        f"Critical Events: {len(state_map.critical_events)} detected\n"
    )

    for event in state_map.critical_events:
        state_map_summary += f"  - [{event.severity.upper()}] {event.event_type} by {event.speaker}: {event.description}\n"

    precedent_section = ""
    if precedent_match is not None:
        gt = precedent_match.ground_truth
        precedent_section = f"""
HISTORICAL PRECEDENT (Similarity: {precedent_match.similarity_score:.3f}):
This is a previously graded call with similar characteristics. Use it for calibration.
- Precedent Call ID: {gt.conversation_id}
- Precedent Call Type: {gt.call_type}
- Precedent Scores: Overall={gt.scores.overall_score}, Compliance={gt.scores.compliance_ethics}, Goal={gt.scores.goal_achievement}, Communication={gt.scores.communication_quality}, Empathy={gt.scores.empathy_tone}, Objection={gt.scores.objection_handling}
- Precedent Outcome: {gt.outcome}
- Precedent Notes: {gt.notes}
- Precedent Compliance Flags: {', '.join(gt.compliance_flags) if gt.compliance_flags else 'None'}

IMPORTANT: Use the precedent scores as a reference point. If the current call is similar in quality, scores should be close. If the current call is better or worse in specific dimensions, adjust scores accordingly with clear reasoning.
"""
    else:
        precedent_section = "\nNo historical precedent found. Grade purely based on the rubric.\n"

    user_prompt = f"""AUDIT THIS DEBT COLLECTION CALL:

METADATA:
- Outstanding Amount: {conversation.metadata.outstanding_amount} INR
- Days Past Due: {conversation.metadata.days_past_due}
- Product Type: {conversation.metadata.product_type}
- Customer Segment: {conversation.metadata.customer_segment}

TRANSCRIPT:
{transcript_text}

STATE MAP ANALYSIS:
{state_map_summary}
{precedent_section}

Produce the final JSON audit result. Be precise in scoring and provide specific evidence from the transcript in your explanations."""

    return user_prompt


async def perform_delta_audit(
    conversation: ConversationInput,
    state_map: StateMap,
    precedent_match: Optional[PrecedentMatch],
) -> AuditResult:
    llm_client = LLMClient.get_instance()

    user_prompt = build_audit_user_prompt(conversation, state_map, precedent_match)

    chat_completion_response = await llm_client.chat.completions.create(
        model=settings.llm_model_name,
        messages=[
            {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    raw_llm_output = chat_completion_response.choices[0].message.content.strip()

    cleaned_output = raw_llm_output
    if cleaned_output.startswith("```"):
        first_newline = cleaned_output.find("\n")
        cleaned_output = cleaned_output[first_newline + 1 :]
    if cleaned_output.endswith("```"):
        cleaned_output = cleaned_output[:-3]
    cleaned_output = cleaned_output.strip()

    parsed_audit_dict = orjson.loads(cleaned_output.encode("utf-8"))

    parsed_audit_dict["conversation_id"] = conversation.conversation_id

    dimensions_raw = parsed_audit_dict.get("dimensions", {})
    validated_dimensions = {}
    for dimension_name, dimension_data in dimensions_raw.items():
        if isinstance(dimension_data, dict):
            validated_dimensions[dimension_name] = AuditDimensionScore(
                score=dimension_data.get("score", 3),
                explanation=dimension_data.get("explanation", ""),
            )

    precedent_id = None
    precedent_similarity = None
    if precedent_match is not None:
        precedent_id = precedent_match.ground_truth.conversation_id
        precedent_similarity = precedent_match.similarity_score

    audit_result = AuditResult(
        conversation_id=conversation.conversation_id,
        overall_score=parsed_audit_dict.get("overall_score", 3),
        dimensions=validated_dimensions,
        outcome=parsed_audit_dict.get("outcome", "refused"),
        summary=parsed_audit_dict.get("summary", ""),
        compliance_flags=parsed_audit_dict.get("compliance_flags", []),
        improvement_suggestions=parsed_audit_dict.get("improvement_suggestions", []),
        sentiment_shifts=parsed_audit_dict.get("sentiment_shifts", []),
        precedent_conversation_id=precedent_id,
        precedent_similarity_score=precedent_similarity,
    )

    return audit_result
