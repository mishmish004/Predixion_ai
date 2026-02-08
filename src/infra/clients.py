import redis.asyncio as aioredis
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from src.core.config import settings


class RedisClient:
    _instance: aioredis.Redis | None = None

    @classmethod
    def get_instance(cls) -> aioredis.Redis:
        if cls._instance is None:
            cls._instance = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
            )
        return cls._instance

    @classmethod
    async def close(cls) -> None:
        if cls._instance is not None:
            await cls._instance.close()
            cls._instance = None


class QdrantSingleton:
    _instance: AsyncQdrantClient | None = None

    @classmethod
    def get_instance(cls) -> AsyncQdrantClient:
        if cls._instance is None:
            cls._instance = AsyncQdrantClient(url=settings.qdrant_url)
        return cls._instance

    @classmethod
    async def close(cls) -> None:
        if cls._instance is not None:
            await cls._instance.close()
            cls._instance = None


class LLMClient:
    _instance: AsyncOpenAI | None = None

    @classmethod
    def get_instance(cls) -> AsyncOpenAI:
        if cls._instance is None:
            cls._instance = AsyncOpenAI(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
            )
        return cls._instance
