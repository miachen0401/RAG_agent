# LangGraph RAG Agent with LLM-based Routing

A **LangGraph-based RAG system** with intelligent LLM-powered routing between three execution paths:

1. **SEMANTIC_QUERY**: Document content questions using RAG
2. **METADATA_QUERY**: Project names, identifiers, and file metadata
3. **DATA_ANALYSIS**: Data visualization and metrics comparison

The system uses **ZHIPU Embedding-3** for semantic search, **ChromaDB** for vector storage, and **GLM-4-Flash** for answer generation.

## Features

- **LLM-based Routing**: GLM-4-AirX classifier automatically routes queries to the appropriate execution path
- **Vector-based RAG**: ZHIPU Embedding-3 (2048-dim) + ChromaDB for semantic similarity search
- **Metadata Extraction**: Direct metadata queries without LLM overhead
- **Modular LLM Configs**: Separate configuration files for each LLM component
- **GLM-4-Flash Integration**: ZHIPU AI LLM for answer generation
- **Production Logging**: Comprehensive logging with configurable levels
- **LangGraph Orchestration**: Robust state management and workflow control

## Architecture

```text
User Query
   |
   v
Router Node (GLM-4-AirX LLM Classification)
   |
   ├── SEMANTIC_QUERY (Document Content)
   |      ├── Vector retrieval (ZHIPU Embedding-3 + ChromaDB)
   |      ├── Context formatting from chunks
   |      └── GLM-4-Flash answer generation
   |
   ├── METADATA_QUERY (Project Names / Identifiers)
   |      ├── Vector search to find matching documents
   |      ├── Extract metadata (file_name, eln_id, etc.)
   |      └── Format response (NO LLM)
   |
   └── DATA_ANALYSIS (Visualization / Metrics)
          └── Placeholder for future implementation
```

## Project Structure

```
.
├── src/
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── router.py           # LLM-based intent classification
│   │   ├── rag_node_new.py     # Semantic query node (RAG)
│   │   ├── metadata_node.py    # Metadata extraction node
│   │   └── analysis_node.py    # Data analysis node (placeholder)
│   ├── llm_client.py           # GLM client wrapper
│   ├── embeddings.py           # ZHIPU Embedding-3 generator
│   ├── vector_retriever.py     # ChromaDB retrieval
│   ├── utils.py                # Config and logging utilities
│   └── main_rag.py             # Main entry point
├── llm_configs/
│   ├── router_config.yaml      # Router LLM configuration
│   ├── rag_config.yaml         # RAG LLM configuration
│   ├── metadata_config.yaml    # Metadata templates
│   ├── data_analysis_config.yaml
│   └── README.md               # LLM configs documentation
├── preprocess/
│   ├── scripts/
│   │   ├── build_rag_index.py  # Build ChromaDB index
│   │   └── inspect_chroma.py   # Inspect ChromaDB contents
│   └── data/                   # Source documents (PDF, etc.)
├── docs/
│   ├── CONFIG_STRUCTURE.md     # Configuration documentation
│   └── Instruction.md          # Design documentation
├── config.yaml                 # Main system configuration
├── pyproject.toml
└── README.md
```

## Installation

This project uses `uv` for Python environment and dependency management.

### Prerequisites

- Python 3.10+
- ZHIPU API key (get from https://open.bigmodel.cn/)

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

### Alternative Installation (pip)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ZHIPU_API_KEY="your_api_key_here"

# Build index and run
python preprocess/scripts/build_rag_index.py
python src/main_rag.py
```

## Usage

### Interactive Mode

Run the agent in interactive mode:

```bash
uv run python src/main_rag.py
```

Example session:
```
RAG System - Interactive Mode (LLM-based Routing)
======================================================================
Ask questions about your documents!
The system will automatically route to:
  - SEMANTIC_QUERY: Document content questions
  - METADATA_QUERY: Project names, ELN IDs, identifiers
  - DATA_ANALYSIS: Data visualization, metrics
======================================================================

Enter your query (or 'quit' to exit): What are the sample preparation methods?

======================================================================
Route: SEMANTIC_QUERY
======================================================================
Response:
======================================================================
Based on the retrieved documents, the sample preparation methods include...
```

### Programmatic Usage

```python
from src.main_rag import initialize_system, run_agent

# Initialize system (load config, create components)
initialize_system()

# Run a semantic query (document content)
result = run_agent("What are the sample preparation methods?")
print(f"Route: {result['route']}")
print(f"Response: {result['response']}")

# Run a metadata query (project names)
result = run_agent("What is the project name for cat eats fish?")
print(f"Response: {result['response']}")

# Run a data analysis query
result = run_agent("Compare project performance metrics")
print(f"Response: {result['response']}")
```

### Example Queries

**SEMANTIC_QUERY** (Document content):
- "What are the sample preparation methods?"
- "Describe the experimental procedure for X"
- "What were the results of the Y experiment?"
- "Explain the methodology used in project Z"

**METADATA_QUERY** (Project names/identifiers):
- "What is the project name for cat eats fish?"
- "Which project has ELN ID 12345?"
- "What is the file name for project X?"
- "Show me the metadata for document Y"

**DATA_ANALYSIS** (Visualization/metrics):
- "Compare project performance metrics"
- "Visualize the results across experiments"
- "Show benchmark comparison"

## Current Implementation

### Router (LLM-based)

**Model**: GLM-4-AirX
**Temperature**: 0.0 (deterministic)
**Config**: `llm_configs/router_config.yaml`

The router uses LLM classification with a structured prompt to categorize user queries into three paths:
- Analyzes query intent using GLM-4-AirX
- Returns one of: `METADATA_QUERY`, `SEMANTIC_QUERY`, `DATA_ANALYSIS`
- Falls back to rule-based routing if LLM fails
- Handles GLM-4.5 "thinking mode" (reasoning_content field)

### SEMANTIC_QUERY Node

**Model**: GLM-4-Flash
**Temperature**: 0.7 (creative)
**Config**: `llm_configs/rag_config.yaml`

Pipeline:
1. **Retrieval**: Vector search using ZHIPU Embedding-3 + ChromaDB
2. **Context Formatting**: Combine top-k chunks with metadata
3. **Generation**: GLM-4-Flash generates answer based on context

### METADATA_QUERY Node

**No LLM used** - Direct metadata extraction
**Config**: `llm_configs/metadata_config.yaml`

Pipeline:
1. **Vector Search**: Find matching documents using partial name matching
2. **Extract Metadata**: file_name, eln_id, project_name, chunk_id, etc.
3. **Format Response**: Template-based formatting (no LLM overhead)

Metadata fields stored in ChromaDB:
- `file_name`: Original filename
- `eln_id`: Electronic lab notebook ID
- `project_name`: Project identifier
- `file_path`: Full path to source file
- `chunk_id`: Unique chunk identifier
- `chunk_index`: Position in document
- `total_chunks`: Total chunks in document
- `preprocessing_date`: When chunk was created

### DATA_ANALYSIS Node

**Status**: Placeholder for future implementation

Planned features:
- Database query integration
- Metrics computation
- Data visualization
- Performance comparison

## Configuration

### Main Configuration (`config.yaml`)

```yaml
embedding:
  model: "embedding-3"
  batch_size: 10
  max_length: 8192

rag:
  chroma_db_path: "data/chroma_db"
  collection_name: "documents"
  top_k: 5

llm_configs:
  directory: "llm_configs"
  router: "router_config.yaml"
  rag: "rag_config.yaml"
  metadata: "metadata_config.yaml"
  data_analysis: "data_analysis_config.yaml"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/rag_system.log"
  console: true
```

### LLM Configurations (`llm_configs/`)

Each LLM component has its own configuration file:

**Router** (`router_config.yaml`):
```yaml
model: "GLM-4-AirX"
temperature: 0.0  # Deterministic classification
max_tokens: 200
system_prompt: |
  You are an intent classification system...
```

**RAG** (`rag_config.yaml`):
```yaml
model: "glm-4.5-flash"
temperature: 0.7  # Creative responses
max_tokens: 2000
system_prompt: |
  You are a knowledgeable research assistant...
```

See `llm_configs/README.md` for detailed documentation on adding new LLM components.

## Inspecting ChromaDB

To view the stored embeddings and metadata:

```bash
uv run python preprocess/scripts/inspect_chroma.py
```

This will display:
- Total number of documents
- Available metadata fields
- Sample chunks with metadata
- Embedding dimensions

## Development

### Adding a New LLM Component

1. Create config file in `llm_configs/my_component_config.yaml`
2. Load config in your component:
   ```python
   from src.utils import load_llm_config

   config = load_llm_config("my_component_config")
   llm_client = GLMClient(
       api_key=api_key,
       model=config.get("model"),
       temperature=config.get("temperature")
   )
   ```
3. Update `config.yaml` to reference the new config

### Temperature Guidelines

- **0.0**: Deterministic (classification, routing)
- **0.3-0.5**: Focused (extraction, summarization)
- **0.7-0.9**: Creative (open-ended generation, RAG)

### Extending the Router

To add new execution paths:

1. Add route to router prompt in `llm_configs/router_config.yaml`
2. Create node class in `src/graph/`
3. Register node in `src/main_rag.py`:
   ```python
   workflow.add_node("NEW_ROUTE", new_node_function)
   workflow.add_conditional_edges("router", route_decision, {
       "SEMANTIC_QUERY": "SEMANTIC_QUERY",
       "METADATA_QUERY": "METADATA_QUERY",
       "DATA_ANALYSIS": "DATA_ANALYSIS",
       "NEW_ROUTE": "NEW_ROUTE"  # Add new route
   })
   ```

## Design Principles

1. **LLM-based routing** for intelligent query classification
2. **Separate configs** for each LLM component (easier tuning)
3. **Skip LLM when possible** (metadata queries use templates)
4. **Explicit state management** with LangGraph
5. **Production-ready logging** at all levels
6. **Modular architecture** for easy extension

## Troubleshooting

### Router returns empty responses

- Check `max_tokens` is sufficient (≥200 for router)
- GLM-4.5 uses `reasoning_content` field - router handles this automatically
- Check logs for `finish_reason: 'length'` (truncated response)

### Embeddings fail

- Verify ZHIPU_API_KEY is set correctly
- Check batch size in config (reduce if rate limited)
- Ensure documents are in correct format

### No chunks retrieved

- Verify ChromaDB path in config
- Run `inspect_chroma.py` to check database contents
- Check embedding model consistency

## Future Enhancements

- [ ] Implement DATA_ANALYSIS node with real database integration
- [ ] Add re-ranking to improve retrieval quality
- [ ] Implement query rewriting for better search
- [ ] Add conversation history and multi-turn support
- [ ] Implement caching for repeated queries
- [ ] Add evaluation metrics for routing accuracy
- [ ] Support multiple document types (Word, Excel, etc.)
- [ ] Add streaming responses for better UX

## License

MIT License

## Contact

For questions or issues, please refer to the documentation in `docs/` or check the LLM configs documentation in `llm_configs/README.md`.
