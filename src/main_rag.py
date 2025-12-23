"""
Main Entry Point for LangGraph RAG Agent with Chunk-based Retrieval.

This module orchestrates the RAG system with:
- Configuration loading
- Logging setup
- Chunk-based similarity search
- GLM-4-Flash LLM integration
"""

import os
import sys
from typing import Dict, Any, Literal
from pathlib import Path

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logging, get_logger
from src.retriever import ChunkRetriever
from src.llm_client import GLMClient
from src.graph.rag_node_new import RAGNode
from src.graph.router import route_query
from src.graph.analysis_node import analysis_node_function


# Global configuration and logger
config = None
logger = None
rag_node_instance = None


class AgentState(TypedDict):
    """
    State schema for the agent graph.

    Attributes:
        query: User input query
        route: Routing decision ('rag' or 'analysis')
        response: Final response to user
        retrieved_chunks: Number of chunks retrieved (RAG path)
        analysis_data: Optional raw analysis data
    """
    query: str
    route: str
    response: str
    retrieved_chunks: int
    analysis_data: Dict[str, Any]


def initialize_system():
    """
    Initialize the RAG system: load config, setup logging, create components.
    """
    global config, logger, rag_node_instance

    # Load configuration
    config = load_config("config.yaml")

    # Setup logging
    logger = setup_logging(config)
    logger.info("="*70)
    logger.info("RAG System Starting")
    logger.info("="*70)

    # Check for API key
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        logger.error("ZHIPU_API_KEY environment variable not set")
        raise ValueError("ZHIPU_API_KEY environment variable is required")

    logger.info("Environment variables loaded")

    # Initialize components
    logger.info("Initializing RAG components")

    # Create retriever
    rag_config = config["rag"]
    retriever = ChunkRetriever(
        chunks_file=rag_config["chunks_file"],
        top_k=rag_config["top_k"],
        min_similarity=rag_config["min_similarity"]
    )

    # Create LLM client
    api_config = config["api"]
    llm_client = GLMClient(
        api_key=api_key,
        model=api_config["model"],
        timeout=api_config["timeout"],
        max_retries=api_config["max_retries"]
    )

    # Create RAG node
    prompts = config["prompts"]
    rag_node_instance = RAGNode(
        retriever=retriever,
        llm_client=llm_client,
        system_prompt=prompts["system_prompt"]
    )

    logger.info("RAG system initialized successfully")


def router_node(state: AgentState) -> AgentState:
    """
    Router node that determines execution path.

    Args:
        state: Current agent state

    Returns:
        Updated state with routing decision
    """
    query = state.get("query", "")
    route = route_query(query)
    state["route"] = route
    logger.info(f"Query routed to: {route}")
    return state


def rag_node_wrapper(state: AgentState) -> AgentState:
    """
    Wrapper for RAG node that uses the global instance.

    Args:
        state: Current agent state

    Returns:
        Updated state with RAG response
    """
    return rag_node_instance.process(state)


def route_decision(state: AgentState) -> Literal["rag", "analysis"]:
    """
    Conditional edge function for routing.

    Args:
        state: Current agent state

    Returns:
        Next node name ('rag' or 'analysis')
    """
    return state["route"]


def create_graph() -> StateGraph:
    """
    Create and configure the LangGraph workflow.

    Graph structure:
        START -> router -> [rag_node | analysis_node] -> END

    Returns:
        Configured StateGraph instance
    """
    logger.info("Creating LangGraph workflow")

    # Initialize graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("rag", rag_node_wrapper)
    workflow.add_node("analysis", analysis_node_function)

    # Set entry point
    workflow.set_entry_point("router")

    # Add conditional routing
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "rag": "rag",
            "analysis": "analysis"
        }
    )

    # Both paths lead to END
    workflow.add_edge("rag", END)
    workflow.add_edge("analysis", END)

    logger.info("LangGraph workflow created")
    return workflow


def run_agent(query: str) -> Dict[str, Any]:
    """
    Run the agent with a user query.

    Args:
        query: User input query string

    Returns:
        Final agent state with response
    """
    logger.info("="*70)
    logger.info(f"Processing query: {query}")
    logger.info("="*70)

    # Create graph
    workflow = create_graph()
    app = workflow.compile()

    # Initialize state
    initial_state = {
        "query": query,
        "route": "",
        "response": "",
        "retrieved_chunks": 0,
        "analysis_data": {}
    }

    # Run graph
    final_state = app.invoke(initial_state)

    logger.info("="*70)
    logger.info("Query processing completed")
    logger.info(f"Route: {final_state['route']}")
    if final_state.get('retrieved_chunks'):
        logger.info(f"Retrieved chunks: {final_state['retrieved_chunks']}")
    logger.info("="*70)

    return final_state


def interactive_mode():
    """
    Run in interactive mode with continuous query input.
    """
    logger.info("Entering interactive mode")

    while True:
        try:
            # Get user input
            user_input = input("\nEnter your query (or 'quit' to exit): ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                logger.info("User requested exit")
                break

            if not user_input:
                continue

            # Run agent
            result = run_agent(user_input)

            # Display response
            print("\n" + "="*70)
            print("Response:")
            print("="*70)
            print(result["response"])
            print("="*70)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}", exc_info=True)
            print(f"\nError: {e}")


def main():
    """
    Main entry point.
    """
    try:
        # Initialize system
        initialize_system()

        # Run interactive mode
        print("\n" + "="*70)
        print("RAG System - Interactive Mode")
        print("="*70)
        print("Ask questions about your documents!")
        print("Type 'quit' to exit")
        print("="*70)

        interactive_mode()

        print("\nGoodbye!")
        logger.info("RAG system shutting down")

    except Exception as e:
        if logger:
            logger.error(f"Fatal error: {e}", exc_info=True)
        else:
            print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
