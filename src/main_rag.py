"""
Main Entry Point for LangGraph RAG Agent with LLM-based Routing.

This module orchestrates the RAG system with:
- LLM-based intent classification
- Three execution paths: SEMANTIC_QUERY, METADATA_QUERY, DATA_ANALYSIS
- ZHIPU Embedding-3 + ChromaDB vector retrieval
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

from src.utils import load_config, setup_logging, get_logger, load_llm_config
from src.embeddings import EmbeddingGenerator
from src.vector_retriever import VectorRetriever
from src.llm_client import GLMClient
from src.graph.rag_node_new import RAGNode
from src.graph.metadata_node import MetadataNode
from src.graph.router import route_query
from src.graph.analysis_node import analysis_node_function


# Global configuration and instances
config = None
logger = None
rag_node_instance = None
metadata_node_instance = None
llm_client_router = None


class AgentState(TypedDict):
    """
    State schema for the agent graph.

    Attributes:
        query: User input query
        route: Routing decision ('SEMANTIC_QUERY', 'METADATA_QUERY', or 'DATA_ANALYSIS')
        response: Final response to user
        retrieved_chunks: Number of chunks retrieved (RAG/Metadata paths)
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
    global config, logger, rag_node_instance, metadata_node_instance, llm_client_router

    # Load configuration
    config = load_config("config.yaml")

    # Setup logging
    logger = setup_logging(config)
    logger.info("="*70)
    logger.info("RAG System Starting - LLM-based Routing")
    logger.info("="*70)

    # Check for API key
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        logger.error("ZHIPU_API_KEY environment variable not set")
        raise ValueError("ZHIPU_API_KEY environment variable is required")

    logger.info("Environment variables loaded")

    # Initialize components
    logger.info("Initializing RAG components")

    # Create embedding generator
    embedding_config = config["embedding"]
    embedding_generator = EmbeddingGenerator(
        api_key=api_key,
        model=embedding_config["model"],
        batch_size=embedding_config["batch_size"],
        max_length=embedding_config["max_length"]
    )

    # Create vector retriever with ChromaDB
    rag_config = config["rag"]
    retriever = VectorRetriever(
        embedding_generator=embedding_generator,
        chroma_db_path=rag_config["chroma_db_path"],
        collection_name=rag_config["collection_name"],
        top_k=rag_config["top_k"]
    )

    # Load LLM configurations
    llm_configs = config.get("llm_configs", {})

    # Load RAG LLM config
    rag_config_file = llm_configs.get("rag", "rag_config.yaml")
    rag_config = load_llm_config(rag_config_file)
    logger.info(f"Loaded RAG config from: {rag_config_file}")

    # Load Router LLM config
    router_config_file = llm_configs.get("router", "router_config.yaml")
    router_config = load_llm_config(router_config_file)
    logger.info(f"Loaded Router config from: {router_config_file}")

    # Create LLM client for RAG
    llm_client = GLMClient(
        api_key=api_key,
        model=rag_config.get("model", "glm-4.5-flash"),
        temperature=rag_config.get("temperature", 0.7),
        timeout=rag_config.get("timeout", 30),
        max_retries=rag_config.get("max_retries", 3)
    )

    # Create LLM client for router
    llm_client_router = GLMClient(
        api_key=api_key,
        model=router_config.get("model", "glm-4.5-flash"),
        temperature=router_config.get("temperature", 0.0),
        timeout=router_config.get("timeout", 30),
        max_retries=router_config.get("max_retries", 3)
    )

    # Create RAG node (SEMANTIC_QUERY path)
    rag_node_instance = RAGNode(
        retriever=retriever,
        llm_client=llm_client,
        system_prompt=rag_config.get("system_prompt", "")
    )

    # Create Metadata node (METADATA_QUERY path)
    metadata_node_instance = MetadataNode(
        retriever=retriever
    )

    logger.info("RAG system initialized successfully")
    logger.info("Routes available: SEMANTIC_QUERY, METADATA_QUERY, DATA_ANALYSIS")


def router_node(state: AgentState) -> AgentState:
    """
    Router node that determines execution path using LLM classification.

    Args:
        state: Current agent state

    Returns:
        Updated state with routing decision
    """
    query = state.get("query", "")
    route = route_query(query, llm_client=llm_client_router)
    state["route"] = route
    logger.info(f"Query routed to: {route}")
    return state


def rag_node_wrapper(state: AgentState) -> AgentState:
    """
    Wrapper for RAG node (SEMANTIC_QUERY path).

    Args:
        state: Current agent state

    Returns:
        Updated state with RAG response
    """
    return rag_node_instance.process(state)


def metadata_node_wrapper(state: AgentState) -> AgentState:
    """
    Wrapper for Metadata node (METADATA_QUERY path).

    Args:
        state: Current agent state

    Returns:
        Updated state with metadata response
    """
    return metadata_node_instance.process(state)


def route_decision(state: AgentState) -> Literal["SEMANTIC_QUERY", "METADATA_QUERY", "DATA_ANALYSIS"]:
    """
    Conditional edge function for routing.

    Args:
        state: Current agent state

    Returns:
        Next node name ('SEMANTIC_QUERY', 'METADATA_QUERY', or 'DATA_ANALYSIS')
    """
    return state["route"]


def create_graph() -> StateGraph:
    """
    Create and configure the LangGraph workflow.

    Graph structure:
        START -> router -> [SEMANTIC_QUERY | METADATA_QUERY | DATA_ANALYSIS] -> END

    Returns:
        Configured StateGraph instance
    """
    logger.info("Creating LangGraph workflow")

    # Initialize graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("SEMANTIC_QUERY", rag_node_wrapper)
    workflow.add_node("METADATA_QUERY", metadata_node_wrapper)
    workflow.add_node("DATA_ANALYSIS", analysis_node_function)

    # Set entry point
    workflow.set_entry_point("router")

    # Add conditional routing
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "SEMANTIC_QUERY": "SEMANTIC_QUERY",
            "METADATA_QUERY": "METADATA_QUERY",
            "DATA_ANALYSIS": "DATA_ANALYSIS"
        }
    )

    # All paths lead to END
    workflow.add_edge("SEMANTIC_QUERY", END)
    workflow.add_edge("METADATA_QUERY", END)
    workflow.add_edge("DATA_ANALYSIS", END)

    logger.info("LangGraph workflow created with 3 execution paths")
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

    print("\nExample queries:")
    print("  - SEMANTIC_QUERY: 'What are the sample preparation methods?'")
    print("  - METADATA_QUERY: 'What is the project name for cat eats fish?'")
    print("  - DATA_ANALYSIS: 'Compare project performance metrics'\n")

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
            print(f"Route: {result['route']}")
            print("="*70)
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
        print("RAG System - Interactive Mode (LLM-based Routing)")
        print("="*70)
        print("Ask questions about your documents!")
        print("The system will automatically route to:")
        print("  - SEMANTIC_QUERY: Document content questions")
        print("  - METADATA_QUERY: Project names, ELN IDs, identifiers")
        print("  - DATA_ANALYSIS: Data visualization, metrics")
        print("\nType 'quit' to exit")
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
