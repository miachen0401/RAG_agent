# LangGraph-based RAG Agent
# LangGraph Agent with RAG + Analysis Routing

This project implements a **LangGraph-based agent system** with explicit routing between two major execution paths:

1. **Document / File Query (RAG-style)**
2. **Analysis / Comparison Tasks (Structured Data)**

The system is intentionally designed to separate **language-uncertain tasks** from **deterministic computation**, enabling easier debugging, iteration, and future productionization.

---

## 1. High-Level Architecture

Overall execution flow:

```text
User Query
   |
   v
Intent Router (LangGraph Node)
   |
   ├── RAG Path (Document / File Query)
   |      ├── Local file loader (initial test)
   |      ├── Retriever (placeholder)
   |      └── Answer synthesis
   |
   └── Analysis Path (Comparison / Metrics)
          ├── Online database (e.g. MonetDB)
          ├── Structured computation (placeholder)
          └── Output / summary
Core design principles:

LangGraph is responsible only for routing and orchestration

RAG and Analysis paths are strictly separated

LLM usage is optional and swappable

Initial implementation favors clarity over intelligence

2. Intent Routing
Current Implementation (Initial Test)
Due to temporary issues with LLM API key access, intent routing is implemented using a simple rule-based if / else approach.

Example:

python
Copy code
if "compare" in user_query or "vs" in user_query:
    route_to = "analysis"
else:
    route_to = "rag"
This allows the graph to function without external dependencies.

Planned Upgrade
The rule-based router will later be replaced with an LLM-based intent classifier.

A placeholder is intentionally kept in the code:

python
Copy code
# TODO: Replace with LLM-based intent classification
# route_to = llm_router(user_query)
This enables seamless reactivation once API access is available.

3. RAG Path (Document / File Query)
Purpose
Handles queries that require retrieving information from documents or files.

Typical examples:

"What does the design document say about X?"

"Find the configuration details for module Y"

Current State (Initial Test)
Documents are loaded from local files

File paths are intentionally left as placeholders

No vector database is required at this stage

python
Copy code
# TODO: Replace with actual local file paths
LOCAL_DOC_PATH = "/path/to/local/files"
This setup allows fast iteration before introducing external storage systems.

Planned Extensions
Replace local loaders with:

Vector databases (FAISS, Chroma, etc.)

Remote document storage

Optional integration with DSPy for:

Query rewriting

Retrieval strategy optimization

Answer synthesis optimization

4. Analysis Path (Comparison / Metrics)
Purpose
Handles queries that require structured comparison or analysis, such as:

"How does project A compare to project B?"

"Is metric X better than the baseline?"

Design Decision
This path is not RAG-based and does not rely on prompt optimization.

Rationale:

Input data is structured

Results must be correct and reproducible

Computation should be deterministic

Current Implementation (Placeholder)
For now, the analysis node simply returns a stub output:

python
Copy code
print("Analysis placeholder: comparison logic will be implemented here.")
This keeps the graph executable while clearly marking unfinished logic.

Planned Extensions
Query structured data from an online database (e.g. MonetDB)

Perform SQL or Pandas-based aggregation

Use an LLM only for explanation or summarization, not computation

5. DSPy Usage Scope
DSPy is intentionally restricted to the RAG path (future work).

Appropriate use cases:

Retrieval strategy optimization

Prompt learning

Multi-hop reasoning

Not appropriate for:

SQL aggregation

Metric comparison

Deterministic analysis

This avoids unnecessary complexity and instability.

6. Suggested Project Structure
text
Copy code
.
├── src/
│   ├── graph/
│   │   ├── router.py        # Intent routing logic
│   │   ├── rag_node.py      # Document query node
│   │   └── analysis_node.py # Analysis placeholder node
│   └── main.py              # Graph entry point
├── docs/
│   └── README.md            # Optional extended design documentation
├── pyproject.toml
└── README.md
7. Environment & Dependency Management (uv)
This project uses uv for Python environment and dependency management.

Create Virtual Environment
bash
Copy code
uv venv
source .venv/bin/activate
Install Dependencies
bash
Copy code
uv pip install langgraph
# optional (future)
# uv pip install dspy
Dependencies should eventually be pinned in pyproject.toml.

8. Development Philosophy
Prefer explicit control flow

Use placeholders instead of premature abstraction

Keep LLM usage optional and replaceable

Separate language reasoning from data computation

This approach maximizes clarity during early development and minimizes refactor cost later.

9. Roadmap
 Replace rule-based router with LLM-based intent classification

 Implement real document loading (local → vector database)

 Integrate DSPy into the RAG path

 Replace analysis placeholder with real comparison logic

 Add routing evaluation and tests