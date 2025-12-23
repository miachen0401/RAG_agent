"""
RAG Node for Document/File Queries with Chunk-based Retrieval.

This module handles queries using:
1. Chunk-based similarity search for retrieval
2. GLM-4-Flash LLM for answer generation
"""

from typing import Dict, Any

from src.retriever import ChunkRetriever
from src.llm_client import GLMClient
from src.utils import get_logger

logger = get_logger(__name__)


class RAGNode:
    """
    RAG Node with chunk-based retrieval and LLM generation.
    """

    def __init__(
        self,
        retriever: ChunkRetriever,
        llm_client: GLMClient,
        system_prompt: str
    ):
        """
        Initialize RAG node.

        Args:
            retriever: ChunkRetriever instance
            llm_client: GLMClient instance
            system_prompt: System prompt for LLM
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.system_prompt = system_prompt

        logger.info("RAGNode initialized with chunk-based retrieval")

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process RAG query through the pipeline.

        Pipeline:
        1. Retrieve similar chunks using similarity search
        2. Format chunks as context
        3. Generate answer using LLM with context

        Args:
            state: Current graph state containing 'query' key

        Returns:
            Updated state with 'response' key
        """
        query = state.get("query", "")
        logger.info(f"Processing RAG query: '{query[:100]}...'")

        try:
            # Step 1: Retrieve similar chunks
            logger.info("Step 1: Retrieving similar chunks")
            chunks_with_scores = self.retriever.retrieve(query)

            if not chunks_with_scores:
                logger.warning("No chunks retrieved for query")
                state["response"] = "I couldn't find relevant information to answer your question."
                state["route"] = "rag"
                return state

            logger.info(f"Retrieved {len(chunks_with_scores)} relevant chunks")

            # Step 2: Format context
            logger.info("Step 2: Formatting context from chunks")
            context = self.retriever.get_context(chunks_with_scores)

            # Step 3: Generate answer
            logger.info("Step 3: Generating answer with LLM")
            answer = self.llm_client.generate(
                query=query,
                context=context,
                system_prompt=self.system_prompt
            )

            if not answer:
                logger.error("Failed to generate answer")
                state["response"] = "I encountered an error while generating the answer. Please try again."
            else:
                logger.info("Successfully generated answer")
                state["response"] = answer

            state["route"] = "rag"
            state["retrieved_chunks"] = len(chunks_with_scores)

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}", exc_info=True)
            state["response"] = f"An error occurred: {str(e)}"
            state["route"] = "rag"

        return state


def rag_node_function(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standalone function for RAG node to be used in LangGraph.

    This is a placeholder - actual instance will be created in main.py with config.

    Args:
        state: Current graph state

    Returns:
        Updated state with RAG response
    """
    # This will be replaced with actual configured instance
    logger.warning("rag_node_function called without proper initialization")
    state["response"] = "RAG node not properly configured"
    state["route"] = "rag"
    return state
