from sentence_transformers import SentenceTransformer
import torch
from typing import List, Union
import numpy as np

class DocumentEmbedder:
    """
    Embedder using BAAI/bge-m3 for legal document embeddings.
    BGE-M3 is excellent for legal text with 8192 token context.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = None,
        normalize: bool = True
    ):
        """
        Initialize the embedder.
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cuda', 'cpu', or None (auto-detect)
            normalize: Whether to normalize embeddings
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.normalize = normalize
        self.model_name = model_name
        
        print(f"Loading {model_name} on {device}...")
        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True
        )
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        instruction: str = None
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            instruction: Optional instruction prefix for queries
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Add instruction prefix for queries (improves retrieval)
        if instruction:
            texts = [f"{instruction}: {text}" for text in texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_query(
        self,
        query: str,
        instruction: str = "Represent this query for retrieving relevant legal documents"
    ) -> np.ndarray:
        """
        Encode a search query with instruction prefix.
        
        Args:
            query: Search query text
            instruction: Query instruction (default for legal docs)
            
        Returns:
            Query embedding
        """
        return self.encode(query, instruction=instruction)[0]
    
    def encode_documents(
        self,
        documents: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode multiple documents for indexing.
        
        Args:
            documents: List of document texts
            batch_size: Processing batch size
            show_progress: Show progress bar
            
        Returns:
            Document embeddings array
        """
        return self.encode(
            documents,
            batch_size=batch_size,
            show_progress=show_progress
        )