"""Qdrant multivector store for ragrag.

Uses Qdrant local on-disk mode (no Docker/server required).
Configured for ColBERT-style MaxSim retrieval with brute-force search
(HnswConfigDiff(m=0)) — optimal for collections < 20k points.
"""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    MultiVectorConfig,
    MultiVectorComparator,
    HnswConfigDiff,
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
    ScoredPoint,
    PointIdsList,
)

from src.models import Segment, MultiVector


COLLECTION_NAME = "ragrag_segments"


class QdrantStore:
    """Local on-disk Qdrant store with multivector (MaxSim) support."""

    def __init__(self, path: str, collection_name: str, embedding_dim: int = 320) -> None:
        """Initialize Qdrant local on-disk store.

        Args:
            path: Directory path for Qdrant on-disk storage.
            collection_name: Name of the Qdrant collection to use.
            embedding_dim: Dimension of each token embedding vector (default 320 for ColQwen3).
        """
        self.client = QdrantClient(path=path)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=HnswConfigDiff(m=0),  # brute-force for <20k points
                ),
            )

    def upsert(self, segment: Segment, vector: MultiVector) -> None:
        """Insert or update a single segment with its multivector embedding."""
        self.upsert_many([(segment, vector)])

    def upsert_many(self, pairs: list[tuple[Segment, MultiVector]]) -> None:
        """Insert or update multiple segments in a single Qdrant call."""
        if not pairs:
            return
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=segment.segment_id,
                    vector=vector,
                    payload=segment.model_dump(),
                )
                for segment, vector in pairs
            ],
        )

    def delete_by_ids(self, point_ids: list[str]) -> None:
        """Delete points by their IDs.

        Args:
            point_ids: List of point IDs (UUIDs as strings) to delete.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=point_ids),
        )

    def search(
        self,
        query_vector: MultiVector,
        top_k: int = 10,
        path_filter: list[str] | None = None,
    ) -> list[ScoredPoint]:
        """Search for similar segments using MaxSim.

        Args:
            query_vector: The multivector query embedding.
            top_k: Maximum number of results to return.
            path_filter: Optional list of file paths to restrict search to.

        Returns:
            List of ScoredPoint results ordered by descending score.
        """
        query_filter: Filter | None = None
        if path_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="path",
                        match=MatchAny(any=path_filter),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )
        return results.points

    def get_collection_info(self) -> dict:
        """Return collection stats.

        Returns:
            Dict with ``points_count`` and ``collection_name``.
        """
        info = self.client.get_collection(self.collection_name)
        return {
            "points_count": info.points_count,
            "collection_name": self.collection_name,
        }
