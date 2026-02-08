import orjson

from src.core.config import settings
from src.core.schema import ConversationInput, StateMap
from src.infra.clients import LLMClient


STATE_EXTRACTION_SYSTEM_PROMPT = """You are an expert analyst for Indian debt collection call auditing.
You will receive a Hinglish (Hindi-English code-mixed) conversation transcript from a collections call.

Your task is to extract a structured "State Map" from the conversation. Analyze every turn carefully.

You MUST respond with ONLY valid JSON matching this exact schema (no markdown, no extra text):
{
  "conversation_id": "string",
  "call_type": "string",
  "intent_flow": [
    {
      "turn_number": integer,
      "speaker": "agent" or "customer",
      "intent": "string - one of: GREETING, IDENTITY_DISCLOSURE, PAYMENT_REMINDER, HARDSHIP_CLAIM, NEGOTIATION, OBJECTION, ACCEPTANCE, THREAT, EMPATHY, INFORMATION_REQUEST, INFORMATION_PROVIDE, PTP_COMMITMENT, CALLBACK_SCHEDULE, CLOSING, REFUSAL, DISPUTE, OTHER",
      "content_summary": "brief English summary of what was said"
    }
  ],
  "critical_events": [
    {
      "event_type": "string - one of: THREAT, EMPATHY, HARDSHIP_CLAIM, PTP, NEGOTIATION, COMPLIANCE_VIOLATION, GREETING, CLOSING, DISPUTE_RAISED, SOLUTION_OFFERED",
      "speaker": "agent" or "customer",
      "description": "English description of the event",
      "severity": "low" or "medium" or "high"
    }
  ],
  "customer_sentiment_trajectory": "string describing how customer sentiment changed through the call, e.g. 'neutral -> frustrated -> accepting'",
  "agent_compliance_summary": "string summarizing whether agent followed RBI guidelines - proper introduction, no threats, no harassment, respectful tone, proper disclosure",
  "negotiation_outcome": "string - what was the negotiation result: PTP with amount/date, partial payment agreed, callback scheduled, customer refused, settlement discussed, etc.",
  "hardship_indicators": ["list of strings - any hardship reasons mentioned: job loss, medical emergency, business loss, etc."],
  "overall_tone": "string - cooperative, hostile, neutral, empathetic, aggressive, professional"
}

Key rules for analysis:
- Hinglish understanding: "dhamki" = threat, "paisa" = money, "EMI" = monthly installment, "PTP" = promise to pay
- Flag any agent behavior that violates RBI Fair Practices Code: threats of physical visit, abusive language, calling at inappropriate hours, harassment, not disclosing identity
- Note if agent offered restructuring options when customer expressed hardship
- Track sentiment shifts carefully - this is critical for quality assessment
- "ghar pe log aayenge" (people will come to your home) is a THREAT and a compliance violation
- Dismissing genuine hardship ("yeh sab bahana hai" = these are all excuses) is a compliance concern"""


def build_transcript_text(conversation: ConversationInput) -> str:
    transcript_lines = []
    for index, message in enumerate(conversation.messages):
        speaker_label = message.role.upper()
        transcript_lines.append(f"Turn {index + 1} [{speaker_label}]: {message.content}")
    return "\n".join(transcript_lines)


async def extract_state_map(conversation: ConversationInput) -> StateMap:
    llm_client = LLMClient.get_instance()

    transcript_text = build_transcript_text(conversation)

    user_prompt = f"""Analyze this Hinglish debt collection call transcript and extract the State Map.

Call Type: {conversation.call_type}
Outstanding Amount: {conversation.metadata.outstanding_amount} INR
Days Past Due: {conversation.metadata.days_past_due}
Product Type: {conversation.metadata.product_type}

TRANSCRIPT:
{transcript_text}

Respond with ONLY the JSON State Map. No markdown formatting, no code blocks, just raw JSON."""

    chat_completion_response = await llm_client.chat.completions.create(
        model=settings.llm_model_name,
        messages=[
            {"role": "system", "content": STATE_EXTRACTION_SYSTEM_PROMPT},
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

    parsed_state_map_dict = orjson.loads(cleaned_output.encode("utf-8"))

    parsed_state_map_dict["conversation_id"] = conversation.conversation_id
    parsed_state_map_dict["call_type"] = conversation.call_type

    state_map = StateMap.model_validate(parsed_state_map_dict)
    return state_map


def serialize_state_map_for_embedding(state_map: StateMap) -> str:
    parts = [
        f"call_type: {state_map.call_type}",
        f"tone: {state_map.overall_tone}",
        f"sentiment: {state_map.customer_sentiment_trajectory}",
        f"compliance: {state_map.agent_compliance_summary}",
        f"outcome: {state_map.negotiation_outcome}",
    ]

    for event in state_map.critical_events:
        parts.append(f"event: {event.event_type} by {event.speaker} severity={event.severity}")

    if state_map.hardship_indicators:
        parts.append(f"hardship: {', '.join(state_map.hardship_indicators)}")

    return " | ".join(parts)
