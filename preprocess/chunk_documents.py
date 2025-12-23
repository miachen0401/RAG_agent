"""
Document Preprocessing and Chunking Script

This script processes all documents in the project_folder directory and creates
token-based chunks with metadata. The chunks are saved to a JSON file for
later use in RAG search.

Usage:
    python preprocess/chunk_documents.py

Output:
    preprocess/chunks.json - All chunks with metadata
    preprocess/chunk_stats.json - Statistics about the chunking process
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocess.chunking import create_chunker
from preprocess.document_loader import DocumentLoader, find_documents_in_folders


class DocumentPreprocessor:
    """
    Main preprocessor for documents.

    Handles the complete pipeline:
    1. Finding documents in sample_documents
    2. Loading and parsing documents
    3. Extracting sections
    4. Chunking with token-based splitting
    5. Saving chunks with metadata
    """

    def __init__(
        self,
        documents_dir: str = "project_folder",
        output_dir: str = "preprocess",
        chunk_size: int = 500,
        overlap: int = 80
    ):
        """
        Initialize the preprocessor.

        Args:
            documents_dir: Directory containing documents to process
            output_dir: Directory to save processed chunks
            chunk_size: Number of tokens per chunk
            overlap: Number of overlapping tokens between chunks
        """
        self.documents_dir = documents_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.chunker = create_chunker(chunk_size=chunk_size, overlap=overlap)
        self.loader = DocumentLoader()

        # Ensure output directory exists
        Path(output_dir).mkdir(exist_ok=True)

    def process_all_documents(self) -> Dict[str, Any]:
        """
        Process all documents and create chunks.

        Returns:
            Dictionary containing all chunks and statistics
        """
        print(f"Searching for documents in: {self.documents_dir}")

        # Find all documents
        document_infos = find_documents_in_folders(self.documents_dir)

        if not document_infos:
            print(f"No documents found in {self.documents_dir}")
            return {
                "chunks": [],
                "stats": {
                    "total_documents": 0,
                    "total_chunks": 0,
                    "error": "No documents found"
                }
            }

        print(f"Found {len(document_infos)} document(s)")

        all_chunks = []
        stats = {
            "total_documents": len(document_infos),
            "total_chunks": 0,
            "total_sections": 0,
            "total_tokens": 0,
            "documents": [],
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "processing_date": datetime.now().isoformat()
        }

        # Process each document
        for doc_info in document_infos:
            folder_name = doc_info["folder_name"]
            file_path = doc_info["file_path"]

            print(f"\nProcessing: {folder_name} ({file_path})")

            try:
                # Load document and extract sections
                doc_data = self.loader.load_document(file_path)

                sections = doc_data["sections"]
                eln_id = doc_data.get("eln_id")

                print(f"  - Found {len(sections)} section(s)")
                if eln_id:
                    print(f"  - ELN ID: {eln_id}")

                # Create base metadata for all chunks from this document
                base_metadata = {
                    "file_name": folder_name,
                    "file_path": file_path,
                    "eln_id": eln_id
                }

                # Chunk all sections
                doc_chunks = self.chunker.chunk_sections(sections, base_metadata)

                print(f"  - Created {len(doc_chunks)} chunk(s)")

                # Calculate total tokens for this document
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
                print(f"  - ERROR: {str(e)}")
                stats["documents"].append({
                    "folder_name": folder_name,
                    "file_path": file_path,
                    "error": str(e)
                })

        stats["total_chunks"] = len(all_chunks)

        return {
            "chunks": all_chunks,
            "stats": stats
        }

    def save_chunks(self, chunks: List[Dict[str, Any]], output_file: str = "chunks.json"):
        """
        Save chunks to JSON file.

        Args:
            chunks: List of chunk dictionaries
            output_file: Output filename (saved in output_dir)
        """
        output_path = Path(self.output_dir) / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(chunks)} chunks to: {output_path}")

    def save_stats(self, stats: Dict[str, Any], output_file: str = "chunk_stats.json"):
        """
        Save processing statistics to JSON file.

        Args:
            stats: Statistics dictionary
            output_file: Output filename (saved in output_dir)
        """
        output_path = Path(self.output_dir) / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Saved statistics to: {output_path}")

    def run(self):
        """
        Run the complete preprocessing pipeline.
        """
        print("=" * 70)
        print("Document Preprocessing and Chunking")
        print("=" * 70)
        print(f"Chunk size: {self.chunk_size} tokens")
        print(f"Overlap: {self.overlap} tokens")
        print()

        # Process all documents
        result = self.process_all_documents()

        chunks = result["chunks"]
        stats = result["stats"]

        # Save results
        if chunks:
            self.save_chunks(chunks)
            self.save_stats(stats)

            # Print summary
            print("\n" + "=" * 70)
            print("Summary")
            print("=" * 70)
            print(f"Total documents processed: {stats['total_documents']}")
            print(f"Total sections extracted: {stats['total_sections']}")
            print(f"Total chunks created: {stats['total_chunks']}")
            print(f"Total tokens processed: {stats['total_tokens']:,}")
            print(f"Average tokens per chunk: {stats['total_tokens'] / stats['total_chunks']:.1f}")
            print("=" * 70)
        else:
            print("\nNo chunks were created. Please check the input directory.")


def main():
    """
    Main entry point.
    """
    preprocessor = DocumentPreprocessor(
        documents_dir="project_folder",
        output_dir="preprocess/chunks",
        chunk_size=500,
        overlap=80
    )
    preprocessor.run()


if __name__ == "__main__":
    main()
