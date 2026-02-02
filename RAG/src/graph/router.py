"""
Intent Router for LangGraph RAG Agent

This module handles routing between different execution paths using LLM-based classification.
Routes:
- METADATA_QUERY: Project names, ELN IDs, identifiers
- SEMANTIC_QUERY: Document content, methods, results
- DATA_ANALYSIS: Data visualization, metrics comparison
"""

from typing import Literal
import os

from src.llm_client import GLMClient
from src.utils import get_logger, load_llm_config

logger = get_logger(__name__)

# Cache router configuration
_router_config_cache = None


def get_router_config():
    """Load and cache router configuration."""
    global _router_config_cache
    if _router_config_cache is None:
        _router_config_cache = load_llm_config("router_config")
    return _router_config_cache


def route_query(user_query: str, llm_client: GLMClient = None) -> Literal["METADATA_QUERY", "SEMANTIC_QUERY", "DATA_ANALYSIS"]:
    """
    Route user query to appropriate execution path using LLM classification.

    Args:
        user_query: The user's input query string
        llm_client: GLMClient instance (if None, will create one)

    Returns:
        "METADATA_QUERY" for identifiers/names
        "SEMANTIC_QUERY" for document content queries
        "DATA_ANALYSIS" for data analysis tasks
    """
    # Initialize LLM client if not provided
    if llm_client is None:
        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key:
            logger.error("ZHIPU_API_KEY not found, falling back to rule-based routing")
            return fallback_route_query(user_query)

        # Load router config
        router_config = get_router_config()

        llm_client = GLMClient(
            api_key=api_key,
            model=router_config.get("model", "glm-4.5-flash"),
            temperature=router_config.get("temperature", 0.0)
        )

    try:
        # Load router configuration
        router_config = get_router_config()
        router_prompt = router_config.get("system_prompt", "")

        if not router_prompt:
            logger.error("Router prompt not found in configuration")
            return fallback_route_query(user_query)

        logger.info(f"Routing query: '{user_query[:100]}...'")

        # Get LLM classification using direct API call
        from zhipuai import ZhipuAI

        api_key = os.getenv("ZHIPU_API_KEY")
        client = ZhipuAI(api_key=api_key)

        messages = [
            {"role": "system", "content": router_prompt},
            {"role": "user", "content": user_query}
        ]

        logger.info(f"Calling router with temperature={router_config.get('temperature', 0.0)}, max_tokens={router_config.get('max_tokens', 200)}")

        api_response = client.chat.completions.create(
            model=router_config.get("model", "glm-4.5-flash"),
            messages=messages,
            temperature=router_config.get("temperature", 0.0),
            max_tokens=router_config.get("max_tokens", 200)
        )

        # Get response - check both content and reasoning_content
        message = api_response.choices[0].message
        response = message.content

        # If content is empty, check reasoning_content (GLM-4.5 thinking mode)
        if not response and hasattr(message, 'reasoning_content') and message.reasoning_content:
            logger.info("Content empty, using reasoning_content")
            response = message.reasoning_content

        logger.info(f"Router LLM raw response: '{response}'")
        logger.info(f"Finish reason: {api_response.choices[0].finish_reason}")

        # Check if response is None or empty
        if not response:
            logger.error("Router LLM returned None or empty response")
            logger.error(f"Full API response: {api_response}")
            return fallback_route_query(user_query)

        # Extract route from response
        route = response.strip().upper()

        # Validate route
        valid_routes = ["METADATA_QUERY", "SEMANTIC_QUERY", "DATA_ANALYSIS"]
        if route not in valid_routes:
            logger.warning(f"Invalid route '{route}' from LLM, attempting to extract")
            # Try to find valid route in response
            for valid_route in valid_routes:
                if valid_route in route:
                    route = valid_route
                    break
            else:
                logger.warning(f"Could not extract valid route, defaulting to SEMANTIC_QUERY")
                route = "SEMANTIC_QUERY"

        logger.info(f"Query routed to: {route}")
        return route

    except Exception as e:
        logger.error(f"Error in LLM routing: {e}, falling back to rule-based routing")
        return fallback_route_query(user_query)


def fallback_route_query(user_query: str) -> Literal["METADATA_QUERY", "SEMANTIC_QUERY", "DATA_ANALYSIS"]:
    """
    Fallback rule-based routing when LLM is unavailable.

    Args:
        user_query: The user's input query string

    Returns:
        Route based on keyword matching
    """
    query_lower = user_query.lower()

    # Load fallback keywords from config
    try:
        router_config = get_router_config()
        fallback_keywords = router_config.get("fallback_keywords", {})

        metadata_keywords = fallback_keywords.get("metadata_query", [])
        analysis_keywords = fallback_keywords.get("data_analysis", [])
    except Exception:
        # Hard-coded fallback if config fails
        metadata_keywords = [
            "project name", "folder name", "eln id", "eln number",
            "identifier", "label", "what is the name", "which project",
            "project is", "called", "file name", "what project"
        ]
        analysis_keywords = [
            "visualize", "visualization", "plot", "chart", "graph",
            "compare metrics", "performance comparison", "benchmark"
        ]

    # Check metadata keywords
    if any(keyword in query_lower for keyword in metadata_keywords):
        logger.info("Fallback routing to: METADATA_QUERY")
        return "METADATA_QUERY"

    # Check analysis keywords
    if any(keyword in query_lower for keyword in analysis_keywords):
        logger.info("Fallback routing to: DATA_ANALYSIS")
        return "DATA_ANALYSIS"

    # Default to semantic query
    logger.info("Fallback routing to: SEMANTIC_QUERY")
    return "SEMANTIC_QUERY"
