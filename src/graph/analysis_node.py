"""
Analysis Node for Comparison/Metrics Tasks

This module handles queries that require structured comparison or analysis.
Designed for deterministic computation, not prompt-based reasoning.
"""

from typing import Dict, Any, List, Optional
import json


class AnalysisNode:
    """
    Node for handling structured analysis and comparison tasks.

    Design Principles:
    - Input data is structured
    - Results must be correct and reproducible
    - Computation should be deterministic
    - LLM usage only for explanation/summarization, not computation
    """

    def __init__(self, db_connection: Optional[Any] = None):
        """
        Initialize Analysis node.

        Args:
            db_connection: Optional database connection (e.g., MonetDB)
        """
        self.db_connection = db_connection

    def query_database(self, query: str) -> List[Dict[str, Any]]:
        """
        Query structured data from database.

        Current Implementation: Placeholder returning mock data
        Future: Connect to MonetDB or other online database

        Args:
            query: User query string

        Returns:
            List of query results as dictionaries
        """
        # TODO: Implement actual database query
        # Example future implementation:
        # sql_query = self.generate_sql(query)
        # results = self.db_connection.execute(sql_query)
        # return results

        # Placeholder: return mock structured data
        print(f"Analysis placeholder: Would query database for '{query}'")

        mock_data = [
            {"project": "A", "metric": "performance", "value": 95.2},
            {"project": "B", "metric": "performance", "value": 87.5},
        ]

        return mock_data

    def perform_comparison(self, data: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Perform structured comparison or aggregation.

        This is deterministic computation using SQL/Pandas, not LLM-based reasoning.

        Args:
            data: Structured data to analyze
            query: User query string

        Returns:
            Comparison results as dictionary
        """
        # TODO: Implement real comparison logic
        # Example future implementation:
        # df = pd.DataFrame(data)
        # result = df.groupby('project').agg({'value': 'mean'})
        # comparison = result.to_dict()
        # return comparison

        # Placeholder: simple comparison
        if len(data) >= 2:
            comparison = {
                "project_a": data[0].get("project", "A"),
                "project_b": data[1].get("project", "B"),
                "metric": data[0].get("metric", "performance"),
                "value_a": data[0].get("value", 0),
                "value_b": data[1].get("value", 0),
                "difference": data[0].get("value", 0) - data[1].get("value", 0),
                "winner": data[0].get("project", "A") if data[0].get("value", 0) > data[1].get("value", 0) else data[1].get("project", "B")
            }
        else:
            comparison = {"error": "Insufficient data for comparison"}

        return comparison

    def generate_summary(self, comparison: Dict[str, Any], query: str) -> str:
        """
        Generate human-readable summary of analysis results.

        This is the ONLY place where an LLM might be used (optional).
        The LLM only formats/explains results, does not compute them.

        Args:
            comparison: Computed comparison results
            query: Original user query

        Returns:
            Human-readable summary string
        """
        # TODO: Optional LLM-based summarization
        # Example future implementation:
        # prompt = f"Explain these results: {json.dumps(comparison)}"
        # summary = llm.invoke(prompt)
        # return summary

        # Current implementation: template-based summary
        if "error" in comparison:
            return f"Analysis Error: {comparison['error']}"

        summary = f"Analysis Results for query: '{query}'\n\n"
        summary += f"Comparison: {comparison.get('project_a')} vs {comparison.get('project_b')}\n"
        summary += f"Metric: {comparison.get('metric')}\n\n"
        summary += f"{comparison.get('project_a')}: {comparison.get('value_a')}\n"
        summary += f"{comparison.get('project_b')}: {comparison.get('value_b')}\n"
        summary += f"Difference: {comparison.get('difference'):+.2f}\n"
        summary += f"Winner: {comparison.get('winner')}\n"

        return summary

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process analysis query through the full pipeline.

        This is the main entry point called by LangGraph.

        Args:
            state: Current graph state containing 'query' key

        Returns:
            Updated state with 'response' key
        """
        query = state.get("query", "")

        # Step 1: Query structured data
        data = self.query_database(query)

        # Step 2: Perform deterministic comparison
        comparison = self.perform_comparison(data, query)

        # Step 3: Generate human-readable summary
        summary = self.generate_summary(comparison, query)

        # Update state
        state["response"] = summary
        state["route"] = "analysis"
        state["analysis_data"] = comparison  # Store raw results for potential further processing

        return state


def analysis_node_function(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standalone function for Analysis node to be used in LangGraph.

    Args:
        state: Current graph state

    Returns:
        Updated state with analysis response
    """
    node = AnalysisNode()
    return node.process(state)
