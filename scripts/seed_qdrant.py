"""
Seed script: Loads ground truth data from sample_conversations.json into Qdrant.

Each conversation's ground truth + a text summary is embedded and stored as a
searchable vector point. This is the "precedent database" that Pass 2 queries.

Usage:
    python -m scripts.seed_qdrant
"""

import asyncio
import json
import sys
import time
import uuid

from qdrant_client.models import Distance, PointStruct, VectorParams

from src.core.config import settings
from src.engine.precedent_retriever import generate_embedding_vector
from src.infra.clients import QdrantSingleton


async def wait_for_qdrant(max_retries: int = 15, delay_seconds: float = 2.0) -> None:
    qdrant_client = QdrantSingleton.get_instance()
    for attempt in range(1, max_retries + 1):
        try:
            await qdrant_client.get_collections()
            print(f"Qdrant is ready (attempt {attempt}).")
            return
        except Exception:
            print(f"Waiting for Qdrant... (attempt {attempt}/{max_retries})")
            await asyncio.sleep(delay_seconds)
    print("ERROR: Qdrant did not become available. Check if the container is running.")
    sys.exit(1)


def build_state_map_text_from_ground_truth(conversation_data: dict) -> str:
    ground_truth = conversation_data.get("ground_truth", {})
    metadata = conversation_data.get("metadata", {})
    call_type = conversation_data.get("call_type", "")
    outcome = ground_truth.get("outcome", "")
    notes = ground_truth.get("notes", "")
    compliance_flags = ground_truth.get("compliance_flags", [])
    overall_score = ground_truth.get("overall_score", 3)

    messages = conversation_data.get("messages", [])
    critical_phrases = []
    for msg in messages:
        content = msg.get("content", "").lower()
        if any(keyword in content for keyword in ["dhamki", "threat", "legal action", "ghar pe", "bahana"]):
            critical_phrases.append(f"{msg['role']}: {msg['content'][:80]}")
        if any(keyword in content for keyword in ["samajh", "understand", "help", "solution"]):
            critical_phrases.append(f"{msg['role']}: {msg['content'][:80]}")

    parts = [
        f"call_type: {call_type}",
        f"outcome: {outcome}",
        f"overall_quality: {'excellent' if overall_score >= 4 else 'poor' if overall_score <= 2 else 'average'}",
        f"days_past_due: {metadata.get('days_past_due', 0)}",
        f"product: {metadata.get('product_type', '')}",
        f"notes: {notes}",
    ]

    if compliance_flags:
        parts.append(f"compliance_issues: {', '.join(compliance_flags)}")

    if critical_phrases:
        parts.append(f"key_phrases: {' | '.join(critical_phrases[:5])}")

    return " | ".join(parts)


async def seed_ground_truths(json_file_path: str) -> None:
    await wait_for_qdrant()

    with open(json_file_path, "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    conversations = data.get("conversations", [])
    if not conversations:
        print("No conversations found in the JSON file.")
        return

    qdrant_client = QdrantSingleton.get_instance()

    collections_response = await qdrant_client.get_collections()
    existing_collection_names = [c.name for c in collections_response.collections]

    if settings.qdrant_collection_name in existing_collection_names:
        await qdrant_client.delete_collection(settings.qdrant_collection_name)
        print(f"Deleted existing collection: {settings.qdrant_collection_name}")

    await qdrant_client.create_collection(
        collection_name=settings.qdrant_collection_name,
        vectors_config=VectorParams(
            size=settings.fastembed_vector_dimension,
            distance=Distance.COSINE,
        ),
    )
    print(f"Created collection: {settings.qdrant_collection_name}")

    points_to_upsert = []

    for conversation_data in conversations:
        ground_truth = conversation_data.get("ground_truth", {})
        metadata = conversation_data.get("metadata", {})
        conversation_id = conversation_data.get("conversation_id", "")

        state_map_text = build_state_map_text_from_ground_truth(conversation_data)
        embedding_vector = generate_embedding_vector(state_map_text)

        payload = {
            "conversation_id": conversation_id,
            "call_type": conversation_data.get("call_type", ""),
            "metadata": metadata,
            "scores": {
                "overall_score": ground_truth.get("overall_score", 3),
                "compliance_ethics": ground_truth.get("compliance_ethics", 3),
                "goal_achievement": ground_truth.get("goal_achievement", 3),
                "communication_quality": ground_truth.get("communication_quality", 3),
                "empathy_tone": ground_truth.get("empathy_tone", 3),
                "objection_handling": ground_truth.get("objection_handling", 3),
            },
            "outcome": ground_truth.get("outcome", ""),
            "compliance_flags": ground_truth.get("compliance_flags", []),
            "notes": ground_truth.get("notes", ""),
            "state_map_text": state_map_text,
        }

        point = PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, conversation_id)),
            vector=embedding_vector,
            payload=payload,
        )
        points_to_upsert.append(point)
        print(f"  Prepared: {conversation_id}")

    await qdrant_client.upsert(
        collection_name=settings.qdrant_collection_name,
        points=points_to_upsert,
    )
    print(f"Upserted {len(points_to_upsert)} ground truth points into Qdrant.")

    await QdrantSingleton.close()
    print("Seeding complete.")


if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else "sample_conversations.json"
    asyncio.run(seed_ground_truths(json_path))
