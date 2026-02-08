"""
Main entry point for the Predixion Debt Audit System.

Modes:
  --enqueue   : Load sample_conversations.json into Redis queue
  --worker    : Start the async worker loop to process the queue
  --batch     : Process all conversations directly (no Redis queue)
  --results   : Fetch and print all results stored in Redis

Usage:
  python main.py --enqueue sample_conversations.json
  python main.py --worker
  python main.py --batch sample_conversations.json
  python main.py --results
"""

import argparse
import asyncio
import json
import sys

import orjson

from src.core.config import settings
from src.core.schema import ConversationInput
from src.infra.clients import RedisClient, QdrantSingleton
from src.workers.orchestrator import run_worker_loop, run_batch_processing


async def enqueue_conversations(json_file_path: str) -> None:
    with open(json_file_path, "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    conversations = data.get("conversations", [])
    redis_client = RedisClient.get_instance()

    for conversation_data in conversations:
        conversation_without_ground_truth = {
            "conversation_id": conversation_data["conversation_id"],
            "call_type": conversation_data["call_type"],
            "messages": conversation_data["messages"],
            "metadata": conversation_data["metadata"],
        }
        serialized = orjson.dumps(conversation_without_ground_truth).decode("utf-8")
        await redis_client.rpush(settings.redis_queue_name, serialized)
        print(f"  Enqueued: {conversation_data['conversation_id']}")

    queue_length = await redis_client.llen(settings.redis_queue_name)
    print(f"Total in queue: {queue_length}")
    await RedisClient.close()


async def run_batch_mode(json_file_path: str) -> None:
    with open(json_file_path, "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    conversations_raw = data.get("conversations", [])
    conversations = []
    for conv_data in conversations_raw:
        conversations.append(
            ConversationInput(
                conversation_id=conv_data["conversation_id"],
                call_type=conv_data["call_type"],
                messages=conv_data["messages"],
                metadata=conv_data["metadata"],
            )
        )

    print(f"Processing {len(conversations)} conversations in batch mode...")
    results = await run_batch_processing(conversations)

    output_list = []
    for result in results:
        result_dict = result.model_dump()
        output_list.append(result_dict)
        print(f"\n{'='*60}")
        print(f"Conversation: {result.conversation_id}")
        print(f"Overall Score: {result.overall_score}/5")
        print(f"Outcome: {result.outcome}")
        for dim_name, dim_score in result.dimensions.items():
            print(f"  {dim_name}: {dim_score.score}/5 - {dim_score.explanation[:80]}...")
        if result.compliance_flags:
            print(f"Compliance Flags: {result.compliance_flags}")
        if result.improvement_suggestions:
            print(f"Suggestions: {result.improvement_suggestions[:2]}")
        print(f"Precedent: {result.precedent_conversation_id} (sim: {result.precedent_similarity_score})")

    output_path = "sample_outputs/batch_results.json"
    with open(output_path, "w", encoding="utf-8") as out_file:
        out_file.write(orjson.dumps(output_list, option=orjson.OPT_INDENT_2).decode("utf-8"))
    print(f"\nResults saved to {output_path}")

    await QdrantSingleton.close()


async def fetch_results() -> None:
    redis_client = RedisClient.get_instance()
    cursor = 0
    all_keys = []

    while True:
        cursor, keys = await redis_client.scan(
            cursor=cursor,
            match=f"{settings.redis_results_prefix}:*",
        )
        all_keys.extend(keys)
        if cursor == 0:
            break

    if not all_keys:
        print("No results found in Redis.")
        await RedisClient.close()
        return

    all_keys.sort()
    results = []

    for key in all_keys:
        raw_value = await redis_client.get(key)
        if raw_value:
            result_dict = orjson.loads(raw_value)
            results.append(result_dict)
            print(f"\n{'='*60}")
            print(f"Key: {key}")
            print(f"  Conversation: {result_dict.get('conversation_id')}")
            print(f"  Overall Score: {result_dict.get('overall_score')}/5")
            print(f"  Outcome: {result_dict.get('outcome')}")

    output_path = "sample_outputs/redis_results.json"
    with open(output_path, "w", encoding="utf-8") as out_file:
        out_file.write(orjson.dumps(results, option=orjson.OPT_INDENT_2).decode("utf-8"))
    print(f"\nAll results saved to {output_path}")

    await RedisClient.close()


async def start_worker() -> None:
    try:
        await run_worker_loop()
    except KeyboardInterrupt:
        print("\nWorker stopped.")
    finally:
        await RedisClient.close()
        await QdrantSingleton.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Predixion Debt Audit System")
    parser.add_argument("--enqueue", type=str, help="Path to JSON file to load into Redis queue")
    parser.add_argument("--worker", action="store_true", help="Start the worker loop")
    parser.add_argument("--batch", type=str, help="Process all conversations in batch mode (no Redis)")
    parser.add_argument("--results", action="store_true", help="Fetch and display all results from Redis")

    args = parser.parse_args()

    if args.enqueue:
        asyncio.run(enqueue_conversations(args.enqueue))
    elif args.worker:
        asyncio.run(start_worker())
    elif args.batch:
        asyncio.run(run_batch_mode(args.batch))
    elif args.results:
        asyncio.run(fetch_results())
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
