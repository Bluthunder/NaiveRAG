from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
from typing import List, Dict, Any, Optional
import numpy as np
from uuid import uuid4

class QdrantVectorStore:
    """
    Qdrant vector database for legal document storage and retrieval.
    """
    
    def __init__(
        self,
        collection_name: str = "legal_documents",
        embedding_dim: int = 1024,  # BGE-M3 dimension
        distance_metric: Distance = Distance.COSINE,
        path: str = "./qdrant_data",  # Local storage path
        host: Optional[str] = None,  # Use for remote Qdrant
        port: Optional[int] = None
    ):
        """
        Initialize Qdrant vector store.
        
        Args:
            collection_name: Name of the collection
            embedding_dim: Dimension of embeddings
            distance_metric: Distance metric (COSINE, EUCLID, DOT)
            path: Path for local storage
            host: Qdrant server host (if using remote)
            port: Qdrant server port (if using remote)
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # Initialize client (local or remote)
        if host and port:
            print(f"Connecting to Qdrant at {host}:{port}")
            self.client = QdrantClient(host=host, port=port)
        else:
            print(f"Using local Qdrant at {path}")
            self.client = QdrantClient(path=path)
        
        # Create collection if it doesn't exist
        self._create_collection(distance_metric)
    
    def _create_collection(self, distance_metric: Distance):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_exists = any(
            col.name == self.collection_name for col in collections
        )
        
        if not collection_exists:
            print(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=distance_metric
                )
            )
        else:
            print(f"Collection {self.collection_name} already exists")
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            embeddings: Document embeddings (2D array)
            metadata: Optional metadata for each document
            ids: Optional custom IDs (generated if not provided)
            
        Returns:
            List of document IDs
        """
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(documents))]
        
        if metadata is None:
            metadata = [{} for _ in range(len(documents))]
        
        # Prepare points for upload
        points = []
        for i, (doc_id, doc, emb, meta) in enumerate(
            zip(ids, documents, embeddings, metadata)
        ):
            payload = {
                "text": doc,
                **meta
            }
            
            point = PointStruct(
                id=doc_id,
                vector=emb.tolist(),
                payload=payload
            )
            points.append(point)
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"Added {len(points)} documents to {self.collection_name}")
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filters: Optional filters for metadata
            
        Returns:
            List of search results with scores and metadata
        """
        # Build filter if provided
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            query_filter = Filter(must=conditions)
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "text": result.payload.get("text", ""),
                "metadata": {
                    k: v for k, v in result.payload.items() 
                    if k != "text"
                }
            })
        
        return formatted_results
    
    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        print(f"Deleted collection: {self.collection_name}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vector_size": info.config.params.vectors.size,
            "points_count": info.points_count,
            "distance": info.config.params.vectors.distance
        }