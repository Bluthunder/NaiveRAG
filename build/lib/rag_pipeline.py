from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path

from src.embedder import DocumentEmbedder
from src.vector_store import QdrantVectorStore
from src.llm_client import OllamaLLM, RAGLLMClient

class LegalRAGPipeline:
    """
    Complete RAG pipeline for legal document Q&A.
    """
    
    def __init__(
        self,
        embedder_model: str = "BAAI/bge-m3",
        llm_model: str = "llama3.1:8b",
        collection_name: str = "legal_documents",
        qdrant_path: str = "./qdrant_data",
        device: str = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embedder_model: Embedding model name
            llm_model: LLM model name for Ollama
            collection_name: Qdrant collection name
            qdrant_path: Path for local Qdrant storage
            device: Device for embeddings ('cuda' or 'cpu')
        """
        print("🚀 Initializing Legal RAG Pipeline...")
        
        # Initialize embedder
        print("\n1. Loading embedding model...")
        self.embedder = DocumentEmbedder(
            model_name=embedder_model,
            device=device
        )
        
        # Initialize vector store
        print("\n2. Connecting to Qdrant...")
        self.vector_store = QdrantVectorStore(
            collection_name=collection_name,
            embedding_dim=self.embedder.embedding_dim,
            path=qdrant_path
        )
        
        # Initialize LLM
        print("\n3. Connecting to Ollama...")
        self.llm = OllamaLLM(model_name=llm_model)
        self.rag_client = RAGLLMClient(self.llm)
        
        print("\n✅ RAG Pipeline initialized successfully!")
    
    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[str]:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
            batch_size: Embedding batch size
            show_progress: Show progress bar
            
        Returns:
            List of document IDs
        """
        print(f"\n📄 Adding {len(documents)} documents...")
        
        # Generate embeddings
        embeddings = self.embedder.encode_documents(
            documents,
            batch_size=batch_size,
            show_progress=show_progress
        )
        
        # Store in vector DB
        doc_ids = self.vector_store.add_documents(
            documents=documents,
            embeddings=embeddings,
            metadata=metadata
        )
        
        return doc_ids
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        score_threshold: float = 0.5,
        filters: Optional[Dict] = None,
        return_sources: bool = True
    ) -> Dict[str, any]:
        """
        Query the RAG system.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score
            filters: Optional metadata filters
            return_sources: Include source documents in response
            
        Returns:
            Dict with answer, sources, and scores
        """
        print(f"\n❓ Query: {question}")
        
        # 1. Encode query
        query_embedding = self.embedder.encode_query(question)
        
        # 2. Retrieve relevant documents
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=filters
        )
        
        if not results:
            return {
                "answer": "I couldn't find any relevant documents to answer your question.",
                "sources": [],
                "scores": []
            }
        
        print(f"📚 Retrieved {len(results)} relevant documents")
        
        # 3. Generate answer using LLM
        contexts = [r["text"] for r in results]
        metadata = [r["metadata"] for r in results]
        
        answer = self.rag_client.generate_answer(
            query=question,
            contexts=contexts,
            metadata=metadata
        )
        
        # 4. Prepare response
        response = {
            "answer": answer,
            "num_sources": len(results)
        }
        
        if return_sources:
            response["sources"] = [
                {
                    "text": r["text"],
                    "score": r["score"],
                    "metadata": r["metadata"]
                }
                for r in results
            ]
        
        return response
    
    def batch_query(
        self,
        questions: List[str],
        **kwargs
    ) -> List[Dict[str, any]]:
        """Process multiple queries."""
        return [self.query(q, **kwargs) for q in questions]
    
    def get_stats(self) -> Dict[str, any]:
        """Get pipeline statistics."""
        return {
            "vector_store": self.vector_store.get_collection_info(),
            "embedder": {
                "model": self.embedder.model_name,
                "dimension": self.embedder.embedding_dim,
                "device": self.embedder.device
            },
            "llm": {
                "model": self.llm.model_name,
                "base_url": self.llm.base_url
            }
        }