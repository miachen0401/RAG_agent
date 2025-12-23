"""
Quick Start Example for LangGraph RAG Agent

This script demonstrates both execution paths:
1. RAG path for document queries
2. Analysis path for comparison tasks
"""

from src.main import run_agent


def main():
    """Run example queries to demonstrate the agent."""

    print("=" * 80)
    print("LangGraph RAG Agent - Example Demonstration")
    print("=" * 80)
    print()

    # Example 1: RAG Path - Document Query
    print("\n" + "=" * 80)
    print("EXAMPLE 1: RAG Path - Document Query")
    print("=" * 80)

    rag_query = "What authentication methods are supported?"
    print(f"\nQuery: {rag_query}\n")

    result1 = run_agent(rag_query)

    # Example 2: Another RAG Path Query
    print("\n" + "=" * 80)
    print("EXAMPLE 2: RAG Path - Configuration Query")
    print("=" * 80)

    rag_query2 = "What are the configuration options for the authentication module?"
    print(f"\nQuery: {rag_query2}\n")

    result2 = run_agent(rag_query2)

    # Example 3: Analysis Path - Comparison Query
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Analysis Path - Comparison Query")
    print("=" * 80)

    analysis_query = "Compare project A vs project B performance"
    print(f"\nQuery: {analysis_query}\n")

    result3 = run_agent(analysis_query)

    # Example 4: Another Analysis Path Query
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Analysis Path - Metric Evaluation")
    print("=" * 80)

    analysis_query2 = "Which model has better metrics?"
    print(f"\nQuery: {analysis_query2}\n")

    result4 = run_agent(analysis_query2)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal queries processed: 4")
    print(f"  - RAG path: 2")
    print(f"  - Analysis path: 2")
    print("\nAll examples completed successfully!")
    print("\nNext steps:")
    print("1. Configure LOCAL_DOC_PATH in src/graph/rag_node.py")
    print("2. Add your own documents to the configured path")
    print("3. Set up database connection for analysis queries")
    print("4. Replace rule-based routing with LLM when API key is available")
    print("\nFor interactive mode, run: uv run python run.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
