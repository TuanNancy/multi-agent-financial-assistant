"""
Embedding Service - Sử dụng BGE-M3 model để tạo embeddings
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
from src.config import EMBEDDING_MODEL, EMBEDDING_DEVICE, DEBUG


class EmbeddingService:
    """
    Service để tạo embeddings từ text sử dụng SentenceTransformer
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Khởi tạo EmbeddingService
        
        Args:
            model_name: Tên model (mặc định từ config)
            device: Device để chạy model (cpu/cuda, mặc định từ config)
        """
        self.model_name = model_name or EMBEDDING_MODEL
        self.device = device or EMBEDDING_DEVICE
        
        if DEBUG:
            print(f"Loading embedding model: {self.model_name} on {self.device}")
        
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            if DEBUG:
                print(f"Embedding model loaded successfully. Dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model {self.model_name}: {str(e)}") from e
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode texts thành embeddings
        
        Args:
            texts: Single text string hoặc list of strings
            normalize_embeddings: Có normalize embeddings về unit vector không
            batch_size: Batch size cho encoding
            show_progress_bar: Hiển thị progress bar không
        
        Returns:
            np.ndarray: Embeddings array với shape (n_texts, embedding_dim)
        """
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=normalize_embeddings,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True
            )
            
            if DEBUG and len(texts) > 1:
                print(f"Encoded {len(texts)} texts into embeddings of shape {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            raise RuntimeError(f"Failed to encode texts: {str(e)}") from e
    
    def encode_single(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        """
        Encode một text duy nhất
        
        Args:
            text: Text string
            normalize_embeddings: Có normalize embeddings không
        
        Returns:
            np.ndarray: Embedding vector với shape (embedding_dim,)
        """
        embeddings = self.encode(text, normalize_embeddings=normalize_embeddings)
        return embeddings[0]  # Return first (and only) embedding
    
    def get_embedding_dimension(self) -> int:
        """
        Lấy dimension của embeddings
        
        Returns:
            int: Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()

