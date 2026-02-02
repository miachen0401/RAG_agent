"""
Metadata Query Node for LangGraph RAG Agent

This module handles metadata queries (project names, ELN IDs, file names).
Uses vector search to find matching documents and returns metadata information.
"""

from typing import Dict, Any

from src.vector_retriever import VectorRetriever
from src.utils import get_logger

logger = get_logger(__name__)


class MetadataNode:
    """
    Metadata query node that returns project metadata information.

    Uses vector search to find relevant documents (handles partial names),
    then returns metadata instead of generating semantic answers.
    """

    def __init__(self, retriever: VectorRetriever):
        """
        Initialize metadata node.

        Args:
            retriever: VectorRetriever instance for finding relevant documents
        """
        self.retriever = retriever
        logger.info("MetadataNode initialized")

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process metadata query.

        Pipeline:
        1. Use vector search to find relevant documents (handles partial names)
        2. Extract unique metadata values
        3. Format and return metadata information

        Args:
            state: Current graph state containing 'query' key

        Returns:
            Updated state with 'response' key containing metadata info
        """
        query = state.get("query", "")
        logger.info(f"Processing metadata query: '{query[:100]}...'")

        try:
            # Step 1: Retrieve relevant chunks using vector search
            logger.info("Step 1: Searching for matching documents")
            chunks_with_scores = self.retriever.retrieve(query)

            if not chunks_with_scores:
                logger.warning("No matching documents found")
                state["response"] = "No matching documents found for your query."
                state["route"] = "METADATA_QUERY"
                return state

            logger.info(f"Found {len(chunks_with_scores)} matching chunks")

            # Step 2: Extract and deduplicate metadata
            logger.info("Step 2: Extracting metadata information")
            metadata_info = self._extract_metadata(chunks_with_scores)

            # Step 3: Format response
            logger.info("Step 3: Formatting metadata response")
            response = self._format_metadata_response(metadata_info, query)

            state["response"] = response
            state["route"] = "METADATA_QUERY"
            state["retrieved_chunks"] = len(chunks_with_scores)

            logger.info("Successfully processed metadata query")

        except Exception as e:
            logger.error(f"Error in metadata query pipeline: {e}", exc_info=True)
            state["response"] = f"An error occurred while retrieving metadata: {str(e)}"
            state["route"] = "METADATA_QUERY"

        return state

    def _extract_metadata(self, chunks_with_scores):
        """
        Extract unique metadata values from chunks.

        Args:
            chunks_with_scores: List of (chunk_dict, score) tuples

        Returns:
            Dictionary with deduplicated metadata
        """
        # Collect metadata from all chunks
        projects = []
        seen_projects = set()

        for chunk, score in chunks_with_scores:
            file_name = chunk.get("file_name", "")
            eln_id = chunk.get("eln_id", "")
            file_path = chunk.get("file_path", "")

            # Use file_name as unique key
            if file_name and file_name not in seen_projects:
                seen_projects.add(file_name)
                projects.append({
                    "file_name": file_name,
                    "eln_id": eln_id,
                    "file_path": file_path,
                    "score": score
                })

        # Sort by relevance score
        projects.sort(key=lambda x: x["score"], reverse=True)

        return {
            "projects": projects,
            "total_matches": len(chunks_with_scores)
        }

    def _format_metadata_response(self, metadata_info, query):
        """
        Format metadata information as a readable response.

        Args:
            metadata_info: Dictionary with metadata
            query: Original user query

        Returns:
            Formatted string response
        """
        projects = metadata_info["projects"]

        if not projects:
            return "No project metadata found."

        # Build response
        lines = []
        lines.append(f"Found {len(projects)} matching project(s):\n")

        for i, project in enumerate(projects, 1):
            lines.append(f"{i}. Project: {project['file_name']}")
            if project['eln_id']:
                lines.append(f"   ELN ID: {project['eln_id']}")
            lines.append(f"   Path: {project['file_path']}")
            lines.append(f"   Relevance: {project['score']:.4f}")
            lines.append("")  # Blank line

        return "\n".join(lines)


def metadata_node_function(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standalone function for metadata node to be used in LangGraph.

    This is a placeholder - actual instance will be created in main.py with config.

    Args:
        state: Current graph state

    Returns:
        Updated state with metadata response
    """
    logger.warning("metadata_node_function called without proper initialization")
    state["response"] = "Metadata node not properly configured"
    state["route"] = "METADATA_QUERY"
    return state
