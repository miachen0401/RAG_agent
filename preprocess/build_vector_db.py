"""
Build Vector Database with Embeddings.

This script:
1. Loads preprocessed chunks from chunks.json
2. Generates embeddings using ZHIPU Embedding-3
3. Stores chunks and embeddings in ChromaDB
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logging, get_logger
from src.embeddings import EmbeddingGenerator


def load_chunks(chunks_file: str) -> List[Dict[str, Any]]:
    """
    Load preprocessed chunks from JSON.

    Args:
        chunks_file: Path to chunks.json

    Returns:
        List of chunk dictionaries
    """
    logger.info(f"Loading chunks from {chunks_file}")

    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks


def build_chromadb(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    collection_name: str,
    chroma_db_path: str
):
    """
    Build ChromaDB collection with chunks and embeddings.

    Args:
        chunks: List of chunk dictionaries
        embeddings: List of embedding vectors
        collection_name: Name of ChromaDB collection
        chroma_db_path: Path to persist ChromaDB
    """
    logger.info(f"Building ChromaDB at {chroma_db_path}")

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=chroma_db_path)

    # Delete existing collection if it exists
    try:
        client.delete_collection(name=collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass

    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    logger.info(f"Created collection: {collection_name}")

    # Prepare data for ChromaDB
    ids = [f"chunk_{chunk['global_chunk_id']}" for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "file_name": chunk.get("file_name", ""),
            "file_path": chunk.get("file_path", ""),
            "section_id": str(chunk.get("section_id", "")),
            "section_name": chunk.get("section_name", ""),
            "eln_id": chunk.get("eln_id", "") or "",
            "token_count": str(chunk.get("token_count", 0)),
            "global_chunk_id": str(chunk.get("global_chunk_id", 0))
        }
        for chunk in chunks
    ]

    # Add to ChromaDB in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))

        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )

        logger.info(f"Added batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}")

    logger.info(f"Successfully added {len(chunks)} chunks to ChromaDB")

    # Verify collection
    count = collection.count()
    logger.info(f"ChromaDB collection count: {count}")

    return collection


def main():
    """
    Main execution function.
    """
    # Load configuration
    config = load_config()

    # Setup logging
    global logger
    logger = setup_logging(config)

    logger.info("="*70)
    logger.info("Building Vector Database")
    logger.info("="*70)

    # Check for API key
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        logger.error("ZHIPU_API_KEY environment variable not set")
        raise ValueError("ZHIPU_API_KEY is required")

    # Load chunks
    chunks_file = "preprocess/chunks.json"
    chunks = load_chunks(chunks_file)

    if not chunks:
        logger.error("No chunks found. Please run chunk_documents.py first")
        return

    # Initialize embedding generator
    embedding_config = config["embedding"]
    logger.info("Initializing embedding generator")

    generator = EmbeddingGenerator(
        api_key=api_key,
        model=embedding_config["model"],
        batch_size=embedding_config["batch_size"],
        max_length=embedding_config["max_length"]
    )

    # Generate embeddings
    logger.info("Generating embeddings for all chunks")
    logger.info("This may take a few minutes...")

    chunks_with_embeddings = generator.generate_chunk_embeddings(chunks)
    embeddings = [chunk["embedding"] for chunk in chunks_with_embeddings]

    # Build ChromaDB
    rag_config = config["rag"]
    collection = build_chromadb(
        chunks=chunks,
        embeddings=embeddings,
        collection_name=rag_config["collection_name"],
        chroma_db_path=rag_config["chroma_db_path"]
    )

    # Summary
    logger.info("="*70)
    logger.info("Vector Database Build Complete")
    logger.info("="*70)
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info(f"Embedding dimension: {len(embeddings[0])}")
    logger.info(f"ChromaDB path: {rag_config['chroma_db_path']}")
    logger.info(f"Collection name: {rag_config['collection_name']}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
