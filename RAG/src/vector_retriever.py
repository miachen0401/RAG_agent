"""
Vector-based Chunk Retriever using ChromaDB and ZHIPU Embeddings.

This module provides functionality to:
- Query ChromaDB with embedding vectors
- Retrieve top-k most similar chunks
- Format results for LLM context
"""

from typing import List, Dict, Any, Tuple
import chromadb

from src.embeddings import EmbeddingGenerator
from src.utils import get_logger

logger = get_logger(__name__)


class VectorRetriever:
    """
    Retrieves relevant chunks using embedding-based similarity search with ChromaDB.
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        chroma_db_path: str,
        collection_name: str,
        top_k: int = 5
    ):
        """
        Initialize the vector retriever.

        Args:
            embedding_generator: EmbeddingGenerator instance for query embeddings
            chroma_db_path: Path to ChromaDB storage
            collection_name: Name of the collection to query
            top_k: Number of top similar chunks to retrieve
        """
        self.embedding_generator = embedding_generator
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        self.top_k = top_k

        logger.info(f"Initializing VectorRetriever with top_k={top_k}")

        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.client.get_collection(name=collection_name)

        count = self.collection.count()
        logger.info(f"Connected to ChromaDB collection '{collection_name}' with {count} chunks")

    def retrieve(self, query: str) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve top-k most similar chunks for a query.

        Args:
            query: User query string

        Returns:
            List of (chunk_dict, similarity_score) tuples, sorted by score descending
        """
        logger.info(f"Retrieving chunks for query: '{query[:100]}...'")

        # Generate query embedding
        logger.debug("Generating query embedding")
        query_embedding = self.embedding_generator.generate_embedding(query)

        # Query ChromaDB
        logger.debug(f"Querying ChromaDB for top {self.top_k} results")
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        chunks_with_scores = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, metadata, distance in zip(documents, metadatas, distances):
            # Convert distance to similarity score (cosine distance to similarity)
            # ChromaDB returns cosine distance, convert to similarity: 1 - distance
            similarity = 1 - distance

            chunk_dict = {
                "text": doc,
                "file_name": metadata.get("file_name", ""),
                "file_path": metadata.get("file_path", ""),
                "section_id": metadata.get("section_id", ""),
                "section_name": metadata.get("section_name", ""),
                "eln_id": metadata.get("eln_id", ""),
                "token_count": metadata.get("token_count", ""),
                "global_chunk_id": metadata.get("global_chunk_id", "")
            }

            chunks_with_scores.append((chunk_dict, similarity))

        logger.info(f"Retrieved {len(chunks_with_scores)} chunks")
        logger.debug(f"Similarity scores: {[score for _, score in chunks_with_scores]}")

        return chunks_with_scores

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
            )

            # Add ELN ID if present
            if chunk.get('eln_id'):
                chunk_text += f"ELN ID: {chunk['eln_id']}\n"

            chunk_text += (
                f"Relevance: {score:.3f}\n"
                f"Content:\n{chunk['text']}\n"
            )

            context_parts.append(chunk_text)

        context = "\n---\n".join(context_parts)
        logger.debug(f"Formatted context with {len(chunks_with_scores)} chunks")

        return context
