"""
Embedding Generation Module using ZHIPU Embedding-3.

This module provides functionality to generate embeddings for text chunks
using ZHIPU AI's Embedding-3 model.
"""

from typing import List, Dict, Any
from zhipuai import ZhipuAI

from src.utils import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """
    Generate embeddings using ZHIPU Embedding-3 model.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "embedding-3",
        batch_size: int = 16,
        max_length: int = 8192
    ):
        """
        Initialize embedding generator.

        Args:
            api_key: ZHIPU AI API key
            model: Embedding model name (default: embedding-3)
            batch_size: Batch size for processing
            max_length: Maximum text length per embedding
        """
        self.model = model
        self.batch_size = batch_size
        self.max_length = max_length

        logger.info(f"Initializing EmbeddingGenerator with model={model}")

        try:
            self.client = ZhipuAI(api_key=api_key)
            logger.info("EmbeddingGenerator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingGenerator: {e}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector as list of floats
        """
        try:
            logger.debug(f"Generating embedding for text (length: {len(text)})")

            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )

            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding with dimension: {len(embedding)}")

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}", exc_info=True)
            raise

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            logger.debug(f"Processing batch {i // self.batch_size + 1}/{(len(texts) + self.batch_size - 1) // self.batch_size}")

            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )

                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

                logger.debug(f"Generated {len(batch_embeddings)} embeddings")

            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch: {e}", exc_info=True)
                raise

        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings

    def generate_chunk_embeddings(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for chunks and add to chunk dictionaries.

        Args:
            chunks: List of chunk dictionaries with 'text' field

        Returns:
            Chunks with added 'embedding' field
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        # Extract texts
        texts = [chunk["text"] for chunk in chunks]

        # Generate embeddings
        embeddings = self.generate_embeddings_batch(texts)

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        logger.info("Successfully added embeddings to all chunks")
        return chunks
