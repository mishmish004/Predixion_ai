from typing import Optional

from fastembed import TextEmbedding
from qdrant_client.models import ScoredPoint

from src.core.config import settings
from src.core.schema import (
    ConversationMetadata,
    GroundTruth,
    GroundTruthScores,
    PrecedentMatch,
)
from src.infra.clients import QdrantSingleton


_embedding_model: Optional[TextEmbedding] = None


def get_embedding_model() -> TextEmbedding:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = TextEmbedding(model_name=settings.fastembed_model_name)
    return _embedding_model


def generate_embedding_vector(text: str) -> list[float]:
    model = get_embedding_model()
    embeddings_generator = model.embed([text])
    embedding_list = list(embeddings_generator)
    return embedding_list[0].tolist()


def convert_scored_point_to_ground_truth(scored_point: ScoredPoint) -> GroundTruth:
    payload = scored_point.payload

    metadata_dict = payload.get("metadata", {})
    scores_dict = payload.get("scores", {})

    return GroundTruth(
        conversation_id=payload.get("conversation_id", ""),
        call_type=payload.get("call_type", ""),
        metadata=ConversationMetadata(**metadata_dict),
        scores=GroundTruthScores(**scores_dict),
        outcome=payload.get("outcome", ""),
        compliance_flags=payload.get("compliance_flags", []),
        notes=payload.get("notes", ""),
        state_map_text=payload.get("state_map_text", ""),
    )


async def retrieve_closest_precedent(
    state_map_text: str,
) -> Optional[PrecedentMatch]:
    qdrant_client = QdrantSingleton.get_instance()

    query_vector = generate_embedding_vector(state_map_text)

    search_results: list[ScoredPoint] = await qdrant_client.search(
        collection_name=settings.qdrant_collection_name,
        query_vector=query_vector,
        limit=settings.qdrant_search_limit,
    )

    if not search_results:
        return None

    best_match = search_results[0]

    ground_truth = convert_scored_point_to_ground_truth(best_match)

    return PrecedentMatch(
        ground_truth=ground_truth,
        similarity_score=best_match.score,
    )
