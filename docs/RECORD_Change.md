# Change Records

## 2025-12-22: RAG System with Chunk-based Retrieval and GLM-4-Flash Integration

### Index
Implemented chunk-based RAG system with TF-IDF similarity search and GLM-4-Flash LLM integration for answering queries based on preprocessed document chunks.

### Components Created

1. **Configuration System** (`config.yaml`)
   - Centralized YAML configuration for all parameters
   - Environment variable support for API keys
   - Configurable: top_k, similarity method, model, prompts, logging

2. **Utilities Module** (`src/utils.py`)
   - `load_config()`: Load YAML config with env variable substitution
   - `setup_logging()`: Configure logging per standards (INFO for pipeline, DEBUG for internals)
   - `get_logger()`: Get namespaced logger instances

3. **Chunk Retriever** (`src/retriever.py`)
   - `ChunkRetriever`: TF-IDF-based similarity search
   - Loads preprocessed chunks from `preprocess/chunks.json`
   - Returns top-k most similar chunks with scores
   - Formats chunks as context for LLM

4. **LLM Client** (`src/llm_client.py`)
   - `GLMClient`: ZHIPU AI API client for GLM-4-Flash
   - `generate()`: Generate answer with query + context
   - HTTP client logs suppressed to DEBUG level per standards

5. **Updated RAG Node** (`src/graph/rag_node_new.py`)
   - Integrated chunk retriever + LLM client
   - Pipeline: retrieve → format context → generate answer
   - Uses logging instead of print statements

6. **New Main Script** (`src/main_rag.py`)
   - System initialization: config → logging → components
   - Interactive query input mode
   - LangGraph integration with router
   - Proper error handling and logging

### Dependencies Added

Added to `pyproject.toml`:
- `pyyaml>=6.0`: Config file parsing
- `scikit-learn>=1.0.0`: TF-IDF vectorization
- `numpy>=1.21.0`: Numerical operations
- `zhipuai>=2.0.0`: ZHIPU AI SDK
- `requests>=2.28.0`: HTTP requests

### Configuration

All parameters now in `config.yaml`:
- `top_k`: Number of chunks to retrieve (default: 5)
- `model`: GLM model name (default: glm-4-flash)
- `similarity_method`: TF-IDF or embedding (current: tfidf)
- `system_prompt`: LLM system prompt
- `logging`: Level, format, file path

### Standards Compliance

✅ No hardcoded parameters - all in config.yaml
✅ No print statements - using logging throughout
✅ HTTP logs at DEBUG level
✅ Pipeline events at INFO level
✅ Minimal, high-signal logging
✅ Common components abstracted (utils, retriever, llm_client)

### Usage

```bash
# Set API key
export ZHIPU_API_KEY="your_api_key_here"

# Install dependencies
uv sync

# Run RAG system
uv run python src/main_rag.py
```

### Files Modified

- `config.yaml` (new)
- `src/utils.py` (new)
- `src/retriever.py` (new)
- `src/llm_client.py` (new)
- `src/graph/rag_node_new.py` (new)
- `src/main_rag.py` (new)
- `pyproject.toml` (updated dependencies)

### Environment Variable Loading

Added automatic `.env` file support:
- Added `python-dotenv>=1.0.0` dependency
- `load_config()` automatically loads `.env` if present
- Users can copy `.env.example` to `.env` and add API key
- No need to export manually - scripts read `.env` automatically

### Next Steps

- Run `uv sync` to install new dependencies
- Copy `.env.example` to `.env` and add your `ZHIPU_API_KEY`
- Test with: `uv run python src/main_rag.py`
- Monitor logs in `logs/rag_system.log`

---

## 2025-12-22: Added .env Auto-loading

### Index
Added python-dotenv for automatic .env file loading - users can store API keys in .env file instead of exporting manually.
