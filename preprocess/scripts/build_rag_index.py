"""
Build RAG Index - Combined Script

This script performs complete RAG preprocessing:
1. Load and chunk documents
2. Generate embeddings with ZHIPU Embedding-3
3. Build and persist ChromaDB vector database

Usage:
    uv run python preprocess/scripts/build_rag_index.py
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import chromadb
from chromadb.config import Settings

# Add parent directories to path
script_dir = Path(__file__).parent
preprocess_dir = script_dir.parent
project_dir = preprocess_dir.parent
sys.path.insert(0, str(project_dir))

from preprocess.utils.chunking import create_chunker
from preprocess.utils.document_loader import DocumentLoader, find_documents_in_folders
from src.utils import load_config, setup_logging, get_logger
from src.embeddings import EmbeddingGenerator


def chunk_documents(documents_dir: str, output_dir: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Step 1: Load and chunk documents.

    Args:
        documents_dir: Directory containing documents
        output_dir: Directory to save chunks
        config: Configuration dictionary

    Returns:
        List of chunk dictionaries
    """
    logger.info("="*70)
    logger.info("STEP 1: Document Chunking")
    logger.info("="*70)

    logger.info(f"Searching for documents in: {documents_dir}")

    # Find documents
    document_infos = find_documents_in_folders(documents_dir)

    if not document_infos:
        logger.error(f"No documents found in {documents_dir}")
        raise ValueError(f"No documents found in {documents_dir}")

    logger.info(f"Found {len(document_infos)} document(s)")

    # Initialize components
    chunking_config = config["chunking"]
    chunker = create_chunker(
        chunk_size=chunking_config["chunk_size"],
        overlap=chunking_config["overlap"]
    )
    loader = DocumentLoader()

    # Process documents
    all_chunks = []
    stats = {
        "total_documents": len(document_infos),
        "total_chunks": 0,
        "total_sections": 0,
        "total_tokens": 0,
        "documents": [],
        "processing_date": datetime.now().isoformat()
    }

    for doc_info in document_infos:
        folder_name = doc_info["folder_name"]
        file_path = doc_info["file_path"]

        logger.info(f"Processing: {folder_name}")

        try:
            # Load document
            doc_data = loader.load_document(file_path)
            sections = doc_data["sections"]
            eln_id = doc_data.get("eln_id")

            logger.info(f"  - Found {len(sections)} section(s)")
            if eln_id:
                logger.info(f"  - ELN ID: {eln_id}")

            # Create base metadata
            base_metadata = {
                "file_name": folder_name,
                "file_path": file_path,
                "eln_id": eln_id
            }

            # Chunk sections
            doc_chunks = chunker.chunk_sections(sections, base_metadata)
            logger.info(f"  - Created {len(doc_chunks)} chunk(s)")

            # Calculate tokens
            doc_tokens = sum(chunk["token_count"] for chunk in doc_chunks)

            all_chunks.extend(doc_chunks)

            # Update stats
            stats["total_sections"] += len(sections)
            stats["total_tokens"] += doc_tokens
            stats["documents"].append({
                "folder_name": folder_name,
                "file_path": file_path,
                "eln_id": eln_id,
                "sections": len(sections),
                "chunks": len(doc_chunks),
                "tokens": doc_tokens
            })

        except Exception as e:
            logger.error(f"  - ERROR: {str(e)}", exc_info=True)
            stats["documents"].append({
                "folder_name": folder_name,
                "file_path": file_path,
                "error": str(e)
            })

    stats["total_chunks"] = len(all_chunks)

    # Save chunks and stats
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    chunks_file = output_path / "chunks.json"
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(all_chunks)} chunks to: {chunks_file}")

    stats_file = output_path / "chunk_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved statistics to: {stats_file}")

    logger.info(f"Total chunks: {len(all_chunks)}")
    logger.info(f"Total tokens: {stats['total_tokens']:,}")

    return all_chunks


def generate_embeddings(chunks: List[Dict[str, Any]], embedding_generator: EmbeddingGenerator) -> List[List[float]]:
    """
    Step 2: Generate embeddings for chunks.

    Args:
        chunks: List of chunk dictionaries
        embedding_generator: EmbeddingGenerator instance

    Returns:
        List of embedding vectors
    """
    logger.info("="*70)
    logger.info("STEP 2: Embedding Generation")
    logger.info("="*70)

    logger.info(f"Generating embeddings for {len(chunks)} chunks")
    logger.info("This may take a few minutes...")

    chunks_with_embeddings = embedding_generator.generate_chunk_embeddings(chunks)
    embeddings = [chunk["embedding"] for chunk in chunks_with_embeddings]

    logger.info(f"Generated {len(embeddings)} embeddings")
    logger.info(f"Embedding dimension: {len(embeddings[0])}")

    return embeddings


def build_vector_db(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    collection_name: str,
    chroma_db_path: str
):
    """
    Step 3: Build ChromaDB vector database.

    Args:
        chunks: List of chunk dictionaries
        embeddings: List of embedding vectors
        collection_name: ChromaDB collection name
        chroma_db_path: Path to persist ChromaDB
    """
    logger.info("="*70)
    logger.info("STEP 3: Building Vector Database")
    logger.info("="*70)

    logger.info(f"Initializing ChromaDB at: {chroma_db_path}")

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=chroma_db_path)

    # Delete existing collection
    try:
        client.delete_collection(name=collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass

    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"Created collection: {collection_name}")

    # Prepare data with unique IDs
    # Use combination of file_name, section_id, and chunk_id to ensure uniqueness
    ids = [
        f"{chunk.get('file_name', 'doc')}_{chunk.get('section_id', 0)}_{chunk.get('chunk_id', 0)}"
        for chunk in chunks
    ]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "file_name": chunk.get("file_name", ""),
            "file_path": chunk.get("file_path", ""),
            "section_id": str(chunk.get("section_id", "")),
            "section_name": chunk.get("section_name", ""),
            "eln_id": chunk.get("eln_id", "") or "",
            "token_count": str(chunk.get("token_count", 0)),
            "chunk_id": str(chunk.get("chunk_id", 0)),
            "global_chunk_id": str(chunk.get("global_chunk_id", 0))
        }
        for chunk in chunks
    ]

    # Add to ChromaDB in batches
    batch_size = 100
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))
        batch_num = i // batch_size + 1

        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )

        logger.info(f"Added batch {batch_num}/{total_batches}")

    # Verify
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
    logger.info("RAG Index Builder - Combined Pipeline")
    logger.info("="*70)

    # Check API key
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        logger.error("ZHIPU_API_KEY environment variable not set")
        raise ValueError("ZHIPU_API_KEY is required")

    logger.info("Environment variables loaded")

    # Configuration
    documents_dir = "project_folder"
    output_dir = "preprocess/output"
    rag_config = config["rag"]
    embedding_config = config["embedding"]

    try:
        # Step 1: Chunk documents
        chunks = chunk_documents(documents_dir, output_dir, config)

        # Step 2: Generate embeddings
        logger.info("Initializing embedding generator")
        embedding_generator = EmbeddingGenerator(
            api_key=api_key,
            model=embedding_config["model"],
            batch_size=embedding_config["batch_size"],
            max_length=embedding_config["max_length"]
        )

        embeddings = generate_embeddings(chunks, embedding_generator)

        # Step 3: Build vector database
        collection = build_vector_db(
            chunks=chunks,
            embeddings=embeddings,
            collection_name=rag_config["collection_name"],
            chroma_db_path=rag_config["chroma_db_path"]
        )

        # Final summary
        logger.info("="*70)
        logger.info("RAG Index Build Complete!")
        logger.info("="*70)
        logger.info(f"Total documents processed: {len(set(c['file_name'] for c in chunks))}")
        logger.info(f"Total chunks created: {len(chunks)}")
        logger.info(f"Embedding dimension: {len(embeddings[0])}")
        logger.info(f"Chunks saved to: {output_dir}/chunks.json")
        logger.info(f"Vector DB saved to: {rag_config['chroma_db_path']}")
        logger.info(f"Collection name: {rag_config['collection_name']}")
        logger.info("="*70)
        logger.info("Ready to run RAG system!")

    except Exception as e:
        logger.error(f"Failed to build RAG index: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
