# Change Records

## 2025-12-23: Removed Obsolete TF-IDF Code

### Index
Removed all obsolete sklearn/TF-IDF code and documentation after migration to ZHIPU Embedding-3 + ChromaDB.

### Files Removed

**Source files:**
- `src/retriever.py` - Old TF-IDF-based ChunkRetriever
- `src/graph/rag_node.py` - Old RAG node using TF-IDF
- `src/main.py` - Old main file
- `example.py` - Obsolete example file

**Documentation:**
- `docs/RAG_USAGE_GUIDE.md` - TF-IDF usage documentation
- `docs/IMPLEMENTATION_SUMMARY.md` - TF-IDF implementation details
- `docs/RAG_METHODS.md` - TF-IDF vs embeddings comparison

### Code Updates

**Fixed `src/graph/rag_node_new.py`:**
- Changed import from `ChunkRetriever` to `VectorRetriever`
- Updated type hint from `ChunkRetriever` to `VectorRetriever`

### Dependencies Removed

- `scikit-learn` - No longer needed (replaced by ZHIPU Embedding-3)

### Verification

✅ No remaining references to:
- `sklearn`
- `TfidfVectorizer`
- `ChunkRetriever`
- `src.retriever`

✅ All Python files now use only:
- `VectorRetriever` (ChromaDB-based retrieval)
- ZHIPU Embedding-3 for embeddings
- Semantic similarity search

---

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

---

## 2025-12-22: Replaced TF-IDF with ZHIPU Embedding-3 and ChromaDB

### Index
Replaced keyword-based TF-IDF with neural embeddings (ZHIPU Embedding-3) and vector database (ChromaDB) for semantic similarity search.

### Changes

**Removed:**
- TF-IDF vectorization (scikit-learn)
- JSON-based chunk storage for retrieval

**Added:**
1. **Embedding Generation** (`src/embeddings.py`)
   - `EmbeddingGenerator`: Generate embeddings using ZHIPU Embedding-3
   - Batch processing support
   - Configurable batch size and max length

2. **Vector Database** (`preprocess/build_vector_db.py`)
   - Generate embeddings for all chunks
   - Store in ChromaDB with cosine similarity
   - Persistent storage for vector index

3. **Vector Retriever** (`src/vector_retriever.py`)
   - `VectorRetriever`: Query ChromaDB with query embeddings
   - Returns top-k most similar chunks
   - Semantic similarity instead of keyword matching

4. **Updated Dependencies**
   - Added `chromadb>=0.4.0`
   - Removed `scikit-learn` (no longer needed)

### Configuration

Updated `config.yaml`:
```yaml
rag:
  top_k: 5
  collection_name: "document_chunks"
  chroma_db_path: "chroma_db"

embedding:
  model: "embedding-3"  # ZHIPU Embedding-3
  batch_size: 16
  max_length: 8192
```

### Workflow

**Preprocessing (offline):**
1. Chunk documents → `chunks.json`
2. Generate embeddings → ZHIPU Embedding-3 API
3. Store in ChromaDB → `chroma_db/`

**Query (online):**
1. User query → Generate query embedding
2. Search ChromaDB → Top-k similar chunks
3. LLM generates answer → Based on retrieved context

### Benefits

- ✅ Semantic understanding (not just keywords)
- ✅ Handles paraphrased queries
- ✅ Understands synonyms and context
- ✅ Better retrieval quality
- ✅ Scalable to large document sets

### Usage

```bash
# 1. Install dependencies
uv sync

# 2. Build RAG index (chunks + embeddings + vector DB)
uv run python preprocess/scripts/build_rag_index.py

# 3. Run RAG system
uv run python src/main_rag.py
```

---

## 2025-12-22: Reorganized Preprocess and Combined Scripts

### Index
Reorganized preprocess directory structure and combined chunking + embedding + vector DB into single script for cleaner workflow.

### Changes

**Directory Structure:**
```
preprocess/
├── scripts/
│   └── build_rag_index.py    # Combined: chunk + embed + store
├── utils/
│   ├── chunking.py            # Moved from preprocess/
│   └── document_loader.py     # Moved from preprocess/
└── output/
    ├── chunks.json            # Generated files
    ├── chunk_stats.json
    └── chroma_db/
```

**Benefits:**
- Single command to build complete RAG index
- Cleaner directory organization
- All generated files in `output/` (gitignored)
- Better separation: scripts vs utilities vs output

**Fixed:**
- Added `sniffio>=1.3.0` dependency (required by zhipuai SDK)
- Updated ChromaDB path to `preprocess/output/chroma_db`

**New Single Command:**
```bash
uv run python preprocess/scripts/build_rag_index.py
```

This replaces the previous two-step process.
