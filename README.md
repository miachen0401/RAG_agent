# LangGraph RAG Agent

A **LangGraph-based agent system** with explicit routing between two major execution paths:

1. **Document / File Query (RAG-style)**
2. **Analysis / Comparison Tasks (Structured Data)**

The system is designed to separate **language-uncertain tasks** from **deterministic computation**, enabling easier debugging, iteration, and future productionization.

## Features

- **Chunk-based RAG**: TF-IDF similarity search on preprocessed document chunks
- **GLM-4-Flash Integration**: ZHIPU AI LLM for answer generation
- **Explicit Routing**: Clear separation between RAG and Analysis paths
- **Configuration-driven**: All parameters in `config.yaml`
- **Production Logging**: INFO-level pipeline events, DEBUG-level internals
- **Modular Architecture**: Easy to extend and customize
- **LangGraph Orchestration**: Robust state management and workflow control

## Architecture

```text
User Query
   |
   v
Intent Router (LangGraph Node)
   |
   ├── RAG Path (Document / File Query)
   |      ├── Chunk-based retrieval (TF-IDF similarity)
   |      ├── Context formatting
   |      └── GLM-4-Flash answer generation
   |
   └── Analysis Path (Comparison / Metrics)
          ├── Database query (placeholder)
          ├── Structured computation
          └── Summary generation
```

## Project Structure

```
.
├── src/
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── router.py        # Intent routing logic
│   │   ├── rag_node.py      # Document query node
│   │   └── analysis_node.py # Analysis node
│   ├── __init__.py
│   └── main.py              # Graph entry point
├── docs/
├── Instruction.md           # Design documentation
├── pyproject.toml
└── README.md
```

## Installation

This project uses `uv` for Python environment and dependency management.

### Quick Start with uv

```bash
# 1. Install all dependencies
uv sync

# 2. Set up environment variables
cp .env.example .env
# Edit .env and add your ZHIPU_API_KEY

# 3. Build RAG index (chunks + embeddings + vector DB)
uv run python preprocess/scripts/build_rag_index.py

# 4. Run RAG system (interactive mode)
uv run python src/main_rag.py
```

This will automatically:
- Create a virtual environment
- Install all dependencies from `pyproject.toml`
- Generate `uv.lock` for reproducible builds

### Alternative Installation

For manual setup or using `pip`:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install langgraph typing-extensions

# Run examples
python example.py
```

## Usage

### Interactive Mode

Run the agent in interactive mode:

```bash
uv run python run.py
```

This will start an interactive session where you can enter queries.

### Programmatic Usage

```python
from src.main import run_agent

# Run a RAG query
result = run_agent("What does the design document say about authentication?")
print(result["response"])

# Run an Analysis query
result = run_agent("Compare project A vs project B performance")
print(result["response"])
```

### Example Queries

**RAG Path** (Document queries):
- "What does the design document say about X?"
- "Find the configuration details for module Y"
- "Show me documentation about Z"

**Analysis Path** (Comparison/metrics):
- "Compare project A vs project B"
- "How does X perform against Y?"
- "Analyze the metrics for Z"
- "Is X better than Y?"

## Current Implementation

### Intent Router (`router.py`)

- **Current**: Rule-based routing using keyword matching
- **Future**: LLM-based intent classification (placeholder ready)

Keywords that trigger Analysis path:
- compare, comparison, vs, versus
- better, worse
- metric, metrics, performance, benchmark
- analyze, analysis, evaluate, evaluation

### RAG Node (`rag_node.py`)

- **Current**: Local file loading from specified directory
- **Future**:
  - Vector database integration (FAISS, Chroma)
  - Semantic similarity search
  - DSPy for retrieval optimization
  - LLM-based answer synthesis

### Analysis Node (`analysis_node.py`)

- **Current**: Mock structured data and simple comparison
- **Future**:
  - Database connection (MonetDB, PostgreSQL)
  - SQL/Pandas-based aggregation
  - Optional LLM for explanation (not computation)

## Development Roadmap

- [ ] Replace rule-based router with LLM-based intent classification
- [ ] Implement real document loading and vector database integration
- [ ] Integrate DSPy into the RAG path for optimization
- [ ] Replace analysis placeholder with real database queries
- [ ] Add comprehensive tests and routing evaluation
- [ ] Add logging and monitoring
- [ ] Implement caching for performance

## Configuration

### Setting Document Path

Edit the `LOCAL_DOC_PATH` in `src/graph/rag_node.py`:

```python
LOCAL_DOC_PATH = "/path/to/your/documents"
```

### Adding Database Connection

Modify the `AnalysisNode` initialization in `src/graph/analysis_node.py`:

```python
# In main.py or your custom runner
from your_db_lib import create_connection

db_conn = create_connection("your_connection_string")
analysis_node = AnalysisNode(db_connection=db_conn)
```

## Design Principles

1. **Prefer explicit control flow** over black-box abstractions
2. **Use placeholders** instead of premature optimization
3. **Keep LLM usage optional** and replaceable
4. **Separate language reasoning** from data computation
5. **Maximize clarity** during early development

## Future Extensions

### DSPy Integration (RAG Path Only)

DSPy is appropriate for:
- Retrieval strategy optimization
- Query rewriting
- Multi-hop reasoning
- Answer synthesis

NOT appropriate for:
- SQL aggregation
- Metric comparison
- Deterministic analysis

### LLM Integration

When LLM API access is available:

1. Update `router.py` to use `llm_router()` function
2. Implement answer synthesis in `rag_node.py`
3. Add optional summarization in `analysis_node.py`

## Testing

```bash
# Run tests (when implemented)
pytest

# Run with coverage
pytest --cov=src tests/
```

## Contributing

1. Keep placeholders clearly marked with `TODO` comments
2. Maintain separation between RAG and Analysis paths
3. Avoid adding LLM dependencies for deterministic computation
4. Update documentation when adding features

## License

MIT License

## Contact

For questions or issues, please refer to the `Instruction.md` file for detailed design documentation.
