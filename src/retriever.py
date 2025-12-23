"""
Chunk Retrieval and Similarity Search Module.

This module provides functionality to:
- Load preprocessed chunks
- Calculate similarity between query and chunks
- Retrieve top-k most similar chunks
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import get_logger

logger = get_logger(__name__)


class ChunkRetriever:
    """
    Retrieves relevant chunks based on similarity search.

    Supports TF-IDF based similarity matching.
    """

    def __init__(self, chunks_file: str, top_k: int = 5, min_similarity: float = 0.0):
        """
        Initialize the retriever.

        Args:
            chunks_file: Path to preprocessed chunks JSON file
            top_k: Number of top similar chunks to retrieve
            min_similarity: Minimum similarity score threshold
        """
        self.chunks_file = chunks_file
        self.top_k = top_k
        self.min_similarity = min_similarity

        logger.info(f"Initializing ChunkRetriever with top_k={top_k}")

        # Load chunks
        self.chunks = self._load_chunks()
        logger.info(f"Loaded {len(self.chunks)} chunks from {chunks_file}")

        # Initialize vectorizer
        self.vectorizer = None
        self.chunk_vectors = None
        self._build_index()

    def _load_chunks(self) -> List[Dict[str, Any]]:
        """
        Load chunks from JSON file.

        Returns:
            List of chunk dictionaries
        """
        try:
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            logger.debug(f"Successfully loaded {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Failed to load chunks from {self.chunks_file}: {e}")
            raise

    def _build_index(self):
        """
        Build TF-IDF index for all chunks.
        """
        logger.info("Building TF-IDF index for chunks")

        # Extract text from all chunks
        texts = [chunk["text"] for chunk in self.chunks]

        # Build TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )

        self.chunk_vectors = self.vectorizer.fit_transform(texts)
        logger.info(f"Built TF-IDF index with {self.chunk_vectors.shape[1]} features")

    def retrieve(self, query: str) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve top-k most similar chunks for a query.

        Args:
            query: User query string

        Returns:
            List of (chunk, similarity_score) tuples, sorted by score descending
        """
        logger.info(f"Retrieving chunks for query: '{query[:100]}...'")

        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])

        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.chunk_vectors)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:self.top_k]

        # Filter by minimum similarity
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= self.min_similarity:
                results.append((self.chunks[idx], score))

        logger.info(f"Retrieved {len(results)} chunks with scores >= {self.min_similarity}")
        logger.debug(f"Scores: {[score for _, score in results]}")

        return results

    def get_context(self, chunks_with_scores: List[Tuple[Dict[str, Any], float]]) -> str:
        """
        Format retrieved chunks into context string for LLM.

        Args:
            chunks_with_scores: List of (chunk, score) tuples

        Returns:
            Formatted context string
        """
        if not chunks_with_scores:
            logger.warning("No chunks to format into context")
            return ""

        context_parts = []
        for i, (chunk, score) in enumerate(chunks_with_scores, 1):
            # Format chunk with metadata
            chunk_text = (
                f"[Document {i}]\n"
                f"Source: {chunk.get('file_name', 'unknown')}\n"
                f"Section: {chunk.get('section_name', 'unknown')}\n"
                f"Relevance: {score:.3f}\n"
                f"Content:\n{chunk['text']}\n"
            )
            context_parts.append(chunk_text)

        context = "\n---\n".join(context_parts)
        logger.debug(f"Formatted context with {len(chunks_with_scores)} chunks")

        return context
