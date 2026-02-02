"""
Token-based Chunking Utility

This module provides deterministic, token-based text chunking with fixed rules.
Text is chunked using a chunk size of 500 tokens with an overlap of 80 tokens
to prevent truncation of file paths, identifiers, and numeric values.
"""

import tiktoken
from typing import List, Dict, Any


class TokenChunker:
    """
    Token-based text chunker with fixed chunk size and overlap.

    Uses tiktoken for deterministic tokenization (cl100k_base encoding).
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 80):
        """
        Initialize the chunker.

        Args:
            chunk_size: Number of tokens per chunk (default: 500)
            overlap: Number of overlapping tokens between chunks (default: 80)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Use cl100k_base encoding (used by GPT-4, GPT-3.5-turbo)
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk text into fixed-size token chunks with overlap.

        Args:
            text: Input text to chunk
            metadata: Metadata to attach to each chunk (file_name, section_id, etc.)

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []

        # Encode text into tokens
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)

        chunks = []
        chunk_id = 0
        start_idx = 0

        while start_idx < total_tokens:
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.chunk_size, total_tokens)

            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode tokens back to text
            chunk_text = self.encoding.decode(chunk_tokens)

            # Create chunk with metadata
            chunk = {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "token_count": len(chunk_tokens),
                "start_token_idx": start_idx,
                "end_token_idx": end_idx,
                **metadata  # Include all provided metadata
            }

            chunks.append(chunk)
            chunk_id += 1

            # Move to next chunk position with overlap
            # If this was the last chunk, break
            if end_idx >= total_tokens:
                break

            # Otherwise, move forward by (chunk_size - overlap)
            start_idx += self.chunk_size - self.overlap

        return chunks

    def chunk_sections(
        self,
        sections: List[Dict[str, Any]],
        base_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Chunk multiple document sections.

        Chunking is applied within each detected document section, but section
        boundaries do not constrain chunk size.

        Args:
            sections: List of section dictionaries with 'section_id', 'section_name', 'text'
            base_metadata: Base metadata to include in all chunks (e.g., file_name, eln_id)

        Returns:
            List of all chunks from all sections
        """
        all_chunks = []
        global_chunk_id = 0

        for section in sections:
            section_id = section.get("section_id", "unknown")
            section_name = section.get("section_name", "unknown")
            section_text = section.get("text", "")

            if not section_text.strip():
                continue

            # Create metadata for this section
            section_metadata = {
                **base_metadata,
                "section_id": section_id,
                "section_name": section_name
            }

            # Chunk the section text
            section_chunks = self.chunk_text(section_text, section_metadata)

            # Update global chunk IDs
            for chunk in section_chunks:
                chunk["global_chunk_id"] = global_chunk_id
                global_chunk_id += 1

            all_chunks.extend(section_chunks)

        return all_chunks


def create_chunker(chunk_size: int = 500, overlap: int = 80) -> TokenChunker:
    """
    Factory function to create a TokenChunker instance.

    Args:
        chunk_size: Number of tokens per chunk (default: 500)
        overlap: Number of overlapping tokens between chunks (default: 80)

    Returns:
        TokenChunker instance
    """
    return TokenChunker(chunk_size=chunk_size, overlap=overlap)
