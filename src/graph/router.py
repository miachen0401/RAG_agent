"""
Intent Router for LangGraph RAG Agent

This module handles routing between RAG and Analysis paths based on user query intent.
Currently uses rule-based routing with placeholder for future LLM-based classification.
"""

from typing import Literal


def route_query(user_query: str) -> Literal["rag", "analysis"]:
    """
    Route user query to appropriate execution path.

    Current Implementation: Rule-based routing using simple keyword matching.
    This allows the graph to function without external LLM dependencies.

    Args:
        user_query: The user's input query string

    Returns:
        "rag" for document/file queries, "analysis" for comparison/metrics tasks
    """
    # TODO: Replace with LLM-based intent classification
    # Example future implementation:
    # route_to = llm_router(user_query)
    # return route_to

    # Current rule-based implementation
    query_lower = user_query.lower()

    # Keywords that suggest analysis/comparison tasks
    analysis_keywords = [
        "compare", "comparison", "vs", "versus", "better", "worse",
        "metric", "metrics", "performance", "benchmark", "analyze",
        "analysis", "evaluate", "evaluation"
    ]

    # Check if query contains analysis keywords
    if any(keyword in query_lower for keyword in analysis_keywords):
        return "analysis"

    # Default to RAG path for document/file queries
    return "rag"


def llm_router(user_query: str) -> Literal["rag", "analysis"]:
    """
    Placeholder for future LLM-based intent classification.

    This function will use an LLM to classify the user's intent more intelligently
    once API access is available.

    Args:
        user_query: The user's input query string

    Returns:
        "rag" for document/file queries, "analysis" for comparison/metrics tasks
    """
    # TODO: Implement LLM-based classification
    # Example implementation:
    # prompt = f"Classify this query as either 'rag' or 'analysis': {user_query}"
    # response = llm.invoke(prompt)
    # return response.content

    raise NotImplementedError("LLM-based routing not yet implemented. Using rule-based routing.")
