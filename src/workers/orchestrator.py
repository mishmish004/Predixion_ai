import asyncio
import logging
import sys

import orjson

from src.core.config import settings
from src.core.schema import ConversationInput, AuditResult
from src.engine.state_extractor import extract_state_map, serialize_state_map_for_embedding
from src.engine.precedent_retriever import retrieve_closest_precedent
from src.engine.auditor import perform_delta_audit
from src.infra.clients import RedisClient, QdrantSingleton


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("orchestrator")


async def wait_for_qdrant(max_retries: int = 15, delay_seconds: float = 2.0) -> None:
    qdrant_client = QdrantSingleton.get_instance()
    for attempt in range(1, max_retries + 1):
        try:
            await qdrant_client.get_collections()
            logger.info("Qdrant is ready (attempt %d).", attempt)
            return
        except Exception:
            logger.info("Waiting for Qdrant... (attempt %d/%d)", attempt, max_retries)
            await asyncio.sleep(delay_seconds)
    logger.error("Qdrant did not become available.")
    sys.exit(1)


async def process_single_conversation(conversation: ConversationInput) -> AuditResult:
    logger.info("Pass 1 - State Extraction for %s", conversation.conversation_id)
    state_map = await extract_state_map(conversation)
    logger.info(
        "State Map extracted: %d intents, %d critical events",
        len(state_map.intent_flow),
        len(state_map.critical_events),
    )

    logger.info("Pass 2 - Precedent Retrieval for %s", conversation.conversation_id)
    state_map_text = serialize_state_map_for_embedding(state_map)
    precedent_match = await retrieve_closest_precedent(state_map_text)

    if precedent_match is not None:
        logger.info(
            "Precedent found: %s (similarity: %.3f)",
            precedent_match.ground_truth.conversation_id,
            precedent_match.similarity_score,
        )
    else:
        logger.info("No precedent found, grading without calibration")

    logger.info("Pass 3 - Delta Audit for %s", conversation.conversation_id)
    audit_result = await perform_delta_audit(conversation, state_map, precedent_match)
    logger.info(
        "Audit complete: overall_score=%d, outcome=%s",
        audit_result.overall_score,
        audit_result.outcome,
    )

    return audit_result


async def save_audit_result_to_redis(audit_result: AuditResult) -> None:
    redis_client = RedisClient.get_instance()
    result_key = f"{settings.redis_results_prefix}:{audit_result.conversation_id}"
    serialized_result = orjson.dumps(audit_result.model_dump()).decode("utf-8")
    await redis_client.set(result_key, serialized_result)
    logger.info("Result saved to Redis: %s", result_key)


async def run_worker_loop() -> None:
    await wait_for_qdrant()
    redis_client = RedisClient.get_instance()
    logger.info("Worker started. Polling queue: %s", settings.redis_queue_name)

    while True:
        queue_item = await redis_client.lpop(settings.redis_queue_name)

        if queue_item is None:
            remaining = await redis_client.llen(settings.redis_queue_name)
            if remaining == 0:
                logger.info("Queue is empty. Checking again...")
            await asyncio.sleep(settings.worker_poll_interval_seconds)
            continue

        try:
            conversation_dict = orjson.loads(queue_item)
            conversation = ConversationInput.model_validate(conversation_dict)

            audit_result = await process_single_conversation(conversation)
            await save_audit_result_to_redis(audit_result)

        except Exception as processing_error:
            logger.error(
                "Failed to process conversation: %s",
                str(processing_error),
                exc_info=True,
            )


async def run_batch_processing(conversations: list[ConversationInput]) -> list[AuditResult]:
    await wait_for_qdrant()
    all_results = []
    for conversation in conversations:
        try:
            audit_result = await process_single_conversation(conversation)
            all_results.append(audit_result)
        except Exception as processing_error:
            logger.error(
                "Failed to process %s: %s",
                conversation.conversation_id,
                str(processing_error),
                exc_info=True,
            )
    return all_results
