"""
Inspect ChromaDB Vector Database

This script allows you to view the contents of ChromaDB including:
- Total number of documents
- Sample documents with metadata
- All metadata fields
- Document statistics

Usage:
    uv run python preprocess/scripts/inspect_chroma.py
"""

import sys
from pathlib import Path
import chromadb
from pprint import pprint

# Add project root to path
project_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_dir))

from src.utils import load_config, get_logger

logger = get_logger(__name__)


def inspect_chromadb(chroma_db_path: str, collection_name: str, num_samples: int = 5):
    """
    Inspect ChromaDB database contents.

    Args:
        chroma_db_path: Path to ChromaDB directory
        collection_name: Collection name
        num_samples: Number of sample documents to display
    """
    print("=" * 80)
    print("ChromaDB Inspector")
    print("=" * 80)

    # Connect to ChromaDB
    print(f"\nConnecting to ChromaDB at: {chroma_db_path}")
    client = chromadb.PersistentClient(path=chroma_db_path)

    # Get collection
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Collection: {collection_name}")
    except Exception as e:
        print(f"Error: Collection '{collection_name}' not found")
        print(f"Available collections: {[c.name for c in client.list_collections()]}")
        return

    # Get total count
    total_count = collection.count()
    print(f"Total documents: {total_count}")

    # Get all data (for analysis)
    print("\nFetching all documents...")
    all_data = collection.get(
        include=["documents", "metadatas", "embeddings"]
    )

    # Analyze metadata fields
    print("\n" + "=" * 80)
    print("Metadata Fields")
    print("=" * 80)

    if all_data['metadatas']:
        # Get all unique metadata keys
        all_keys = set()
        for metadata in all_data['metadatas']:
            all_keys.update(metadata.keys())

        print(f"\nMetadata fields stored in ChromaDB:")
        for key in sorted(all_keys):
            print(f"  - {key}")

        # Count unique values for each field
        print("\nMetadata statistics:")

        # File names
        file_names = set(m.get('file_name', '') for m in all_data['metadatas'])
        print(f"\n  Unique file_name values: {len(file_names)}")
        for fname in sorted(file_names):
            count = sum(1 for m in all_data['metadatas'] if m.get('file_name') == fname)
            print(f"    - {fname}: {count} chunks")

        # ELN IDs
        eln_ids = set(m.get('eln_id', '') for m in all_data['metadatas'] if m.get('eln_id'))
        if eln_ids:
            print(f"\n  Unique eln_id values: {len(eln_ids)}")
            for eln_id in sorted(eln_ids):
                count = sum(1 for m in all_data['metadatas'] if m.get('eln_id') == eln_id)
                print(f"    - {eln_id}: {count} chunks")

        # Sections
        section_names = set(m.get('section_name', '') for m in all_data['metadatas'] if m.get('section_name'))
        print(f"\n  Unique section_name values: {len(section_names)}")
        if len(section_names) <= 20:
            for section in sorted(section_names):
                count = sum(1 for m in all_data['metadatas'] if m.get('section_name') == section)
                print(f"    - {section}: {count} chunks")

    # Display sample documents
    print("\n" + "=" * 80)
    print(f"Sample Documents (showing {min(num_samples, total_count)})")
    print("=" * 80)

    sample_data = collection.get(
        limit=num_samples,
        include=["documents", "metadatas"]
    )

    for i, (doc_id, document, metadata) in enumerate(zip(
        sample_data['ids'],
        sample_data['documents'],
        sample_data['metadatas']
    ), 1):
        print(f"\n--- Document {i} ---")
        print(f"ID: {doc_id}")
        print(f"\nMetadata:")
        pprint(metadata, indent=2, width=80)
        print(f"\nText preview (first 200 chars):")
        print(document[:200] + "..." if len(document) > 200 else document)

    # Embedding info
    if all_data['embeddings'] is not None and len(all_data['embeddings']) > 0:
        embedding_dim = len(all_data['embeddings'][0])
        print("\n" + "=" * 80)
        print("Embedding Information")
        print("=" * 80)
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Total embeddings: {len(all_data['embeddings'])}")

    # Test query
    print("\n" + "=" * 80)
    print("Test Query")
    print("=" * 80)
    print("\nTesting similarity search with query: 'test query'")

    from src.embeddings import EmbeddingGenerator
    import os

    api_key = os.getenv("ZHIPU_API_KEY")
    if api_key:
        embedding_gen = EmbeddingGenerator(api_key=api_key, model="embedding-3")
        query_embedding = embedding_gen.generate_embedding("test query")

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        print("\nTop 3 results:")
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            print(f"\n  Result {i}:")
            print(f"    Distance: {dist:.4f}")
            print(f"    File: {meta.get('file_name', 'N/A')}")
            print(f"    Section: {meta.get('section_name', 'N/A')}")
            print(f"    ELN ID: {meta.get('eln_id', 'N/A')}")
            print(f"    Text: {doc[:100]}...")
    else:
        print("ZHIPU_API_KEY not found - skipping query test")

    print("\n" + "=" * 80)


def main():
    """Main execution."""
    # Load config
    config = load_config()
    rag_config = config["rag"]

    chroma_db_path = rag_config["chroma_db_path"]
    collection_name = rag_config["collection_name"]

    # Inspect database
    inspect_chromadb(chroma_db_path, collection_name, num_samples=5)


if __name__ == "__main__":
    main()
