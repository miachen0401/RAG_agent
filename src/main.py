"""
Main Entry Point for LangGraph RAG Agent

This module orchestrates the agent system with explicit routing between
RAG and Analysis execution paths using LangGraph.
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from src.graph.router import route_query
from src.graph.rag_node import rag_node_function
from src.graph.analysis_node import analysis_node_function


class AgentState(TypedDict):
    """
    State schema for the agent graph.

    Attributes:
        query: User input query
        route: Routing decision ('rag' or 'analysis')
        response: Final response to user
        analysis_data: Optional raw analysis data
    """
    query: str
    route: str
    response: str
    analysis_data: Dict[str, Any]


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
    print(f"[Router] Query routed to: {route}")
    return state


def route_decision(state: AgentState) -> Literal["rag", "analysis"]:
    """
    Conditional edge function for routing.

    This determines which node to execute next based on routing decision.

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
    # Initialize graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("rag", rag_node_function)
    workflow.add_node("analysis", analysis_node_function)

    # Set entry point
    workflow.set_entry_point("router")

    # Add conditional routing from router to execution paths
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "rag": "rag",
            "analysis": "analysis"
        }
    )

    # Both execution paths lead to END
    workflow.add_edge("rag", END)
    workflow.add_edge("analysis", END)

    return workflow


def run_agent(query: str) -> Dict[str, Any]:
    """
    Run the agent with a user query.

    Args:
        query: User input query string

    Returns:
        Final agent state with response
    """
    # Create graph
    workflow = create_graph()
    app = workflow.compile()

    # Initialize state
    initial_state = {
        "query": query,
        "route": "",
        "response": "",
        "analysis_data": {}
    }

    # Run graph
    print(f"\n{'='*60}")
    print(f"Processing query: {query}")
    print(f"{'='*60}\n")

    final_state = app.invoke(initial_state)

    print(f"\n{'='*60}")
    print("Response:")
    print(f"{'='*60}\n")
    print(final_state["response"])
    print(f"\n{'='*60}\n")

    return final_state


def main():
    """
    Main function with example usage.
    """
    print("LangGraph RAG Agent - Interactive Mode\n")

    # Example queries
    example_queries = [
        "What does the design document say about authentication?",
        "Compare project A vs project B performance",
        "Find configuration details for the database module",
        "Which model performs better on the benchmark?"
    ]

    print("Example queries:")
    for i, q in enumerate(example_queries, 1):
        print(f"{i}. {q}")
    print()

    # Interactive loop
    while True:
        try:
            user_input = input("Enter your query (or 'quit' to exit): ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not user_input:
                print("Please enter a valid query.\n")
                continue

            # Run agent
            run_agent(user_input)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
