# Change Records

## 2025-12-23: Reorganized to LLM Configs Directory Structure

### Index
Reorganized configuration structure from single `prompts/prompts.yaml` to individual LLM config files in `llm_configs/` directory for better separation and management.

### Changes

**Created `llm_configs/` directory with separate config files:**
- `router_config.yaml` - Router LLM configuration
- `rag_config.yaml` - RAG LLM configuration
- `metadata_config.yaml` - Metadata query templates
- `data_analysis_config.yaml` - Data analysis (placeholder)
- `README.md` - Comprehensive documentation

**Updated Code:**
- `src/utils.py` - Added `load_llm_config()` function
- `src/graph/router.py` - Loads from `router_config.yaml`
- `src/main_rag.py` - Loads from `rag_config.yaml`
- `config.yaml` - Updated to reference `llm_configs` directory

**Removed:**
- `prompts/` directory (replaced by `llm_configs/`)

### New Structure

**Before:**
```
prompts/
└── prompts.yaml
    ├── router: {...}
    ├── semantic_query: {...}
    ├── metadata_query: {...}
    └── data_analysis: {...}
```

**After:**
```
llm_configs/
├── router_config.yaml
├── rag_config.yaml
├── metadata_config.yaml
├── data_analysis_config.yaml
└── README.md
```

### Benefits

- ✅ **Clear Separation:** Each LLM component has its own config file
- ✅ **Easy Management:** Edit individual configs without affecting others
- ✅ **Scalable:** Easy to add new LLM components (just add new .yaml file)
- ✅ **Better Version Control:** Smaller diffs, easier to track changes
- ✅ **Self-Documenting:** File names clearly indicate purpose

### Usage

**Loading LLM Configs:**
```python
from src.utils import load_llm_config

# Load router config
router_config = load_llm_config("router_config")

# Load RAG config
rag_config = load_llm_config("rag_config")

# Access settings
system_prompt = rag_config["system_prompt"]
temperature = rag_config["temperature"]
```

**Adding New LLM Component:**
1. Create `llm_configs/new_component_config.yaml`
2. Define model, temperature, max_tokens, prompts
3. Load in code: `config = load_llm_config("new_component_config")`
4. No changes to config.yaml needed!

### Future-Proof

Easy to add new LLM-powered features:
- Translation LLM
- Summarization LLM
- Code generation LLM
- Q&A generation LLM
- Multi-modal LLM (if supported)

Each gets its own config file in `llm_configs/`.

---

## 2025-12-23: Centralized Prompts Configuration System

### Index
Reorganized all LLM prompts into a centralized YAML configuration file for better management, version control, and maintainability.

### Changes

**Created:**
- `prompts/prompts.yaml` - Centralized prompts configuration
  - Router prompts with polished, professional instructions
  - Semantic query prompts (moved from config.yaml)
  - Metadata query response templates
  - Data analysis prompts (placeholder)
  - Error and fallback messages
  - Model parameters (temperature, max_tokens) per route

- `prompts/README.md` - Comprehensive documentation
  - Usage examples and best practices
  - Temperature and max_tokens guidelines
  - Troubleshooting guide
  - Prompt editing workflow

**Updated:**
- `config.yaml` - Simplified prompts section
  - Now only contains reference to `prompts/prompts.yaml`
  - Removed inline prompt strings

- `src/utils.py` - Added `load_prompts()` function
  - Loads and parses prompts YAML file
  - Returns prompts configuration dictionary

- `src/graph/router.py` - Uses new prompts config
  - Loads router prompts from YAML
  - Caches prompts for performance
  - Gets temperature and max_tokens from config

- `src/main_rag.py` - Loads prompts dynamically
  - Reads prompts config file path from config.yaml
  - Passes semantic_query prompts to RAG node
  - Centralized prompt loading

**Removed:**
- `prompts/LLM_router.txt` - Replaced by YAML config

### Prompts Structure

```yaml
router:
  system_prompt: |
    Professional routing instructions...
  temperature: 0.0
  max_tokens: 10

semantic_query:
  system_prompt: |
    Research assistant instructions...
  user_message_template: |
    Context and query formatting...
  temperature: 0.7
  max_tokens: 2000

metadata_query:
  response_template: |
    Structured output format...

data_analysis:
  system_prompt: |
    Data analysis instructions...

fallback_messages:
  routing_error: |
    Error message...
```

### Benefits

- ✅ All prompts in one file (easy to find and edit)
- ✅ Professional, polished prompt engineering
- ✅ Version control friendly (YAML format)
- ✅ Model parameters co-located with prompts
- ✅ Clear separation by route type
- ✅ Comprehensive documentation
- ✅ Easy to A/B test prompt changes
- ✅ No code changes needed to update prompts

### Migration

Old structure (config.yaml):
```yaml
prompts:
  system_prompt: |
    Simple prompt...
```

New structure:
```yaml
# config.yaml
prompts:
  config_file: "prompts/prompts.yaml"

# prompts/prompts.yaml
semantic_query:
  system_prompt: |
    Detailed, professional prompt...
```

---

## 2025-12-23: Implemented LLM-based Routing with Three Execution Paths

### Index
Replaced rule-based routing with LLM-based classification using prompts/LLM_router.txt. Added three execution paths: SEMANTIC_QUERY (RAG), METADATA_QUERY (direct metadata retrieval), and DATA_ANALYSIS (placeholder).

### New Routes

1. **SEMANTIC_QUERY** (replaces "rag")
   - Document content questions
   - Methods, results, conclusions
   - Semantic understanding of text

2. **METADATA_QUERY** (new functionality)
   - Project names, folder names
   - ELN IDs, identifiers
   - Uses vector search + metadata extraction
   - Returns metadata info without LLM generation

3. **DATA_ANALYSIS** (replaces "analysis")
   - Data visualization
   - Metrics comparison
   - Placeholder for future implementation

### Components Created/Updated

**New Files:**
- `src/graph/metadata_node.py` - MetadataNode for METADATA_QUERY path
  - Uses vector search to find matching documents
  - Extracts and returns metadata (file_name, eln_id, file_path)
  - Handles partial name queries

**Updated Files:**
- `src/graph/router.py` - Complete rewrite for LLM-based routing
  - Uses prompts/LLM_router.txt for classification
  - Calls GLM-4-Flash with temperature=0.0 for deterministic routing
  - Fallback to rule-based routing if LLM fails

- `src/graph/rag_node_new.py` - Updated route name from "rag" to "SEMANTIC_QUERY"

- `src/graph/analysis_node.py` - Updated route name from "analysis" to "DATA_ANALYSIS"

- `src/llm_client.py` - Added default temperature parameter
  - Constructor accepts temperature parameter
  - generate() and generate_simple() use default if not specified

- `src/main_rag.py` - Complete integration of new routing system
  - Three node wrappers: rag_node_wrapper, metadata_node_wrapper
  - Separate LLM client for router (temperature=0)
  - Updated graph with three execution paths

### Routing Prompt

Uses `prompts/LLM_router.txt` with:
- Temperature: 0.0 (deterministic)
- Max tokens: 5 (only return route label)
- Returns: METADATA_QUERY, SEMANTIC_QUERY, or DATA_ANALYSIS

### METADATA_QUERY Implementation

Pipeline:
1. **Vector search**: Query → Embedding → Find similar chunks
2. **Extract metadata**: Deduplicate by file_name, collect eln_id, file_path
3. **Format response**: Return metadata info (no LLM generation)

Example query: "What is the project name for cat eats fish?"
Example response:
```
Found 2 matching project(s):

1. Project: 2025_013_TMPIF1096_cateatfish_D210GX0001
   ELN ID: ELN0010425
   Path: project_folder/2025_013_TMPIF1096_cateatfish_D210GX0001/...
   Relevance: 0.8234
```

### Benefits

- ✅ LLM-based intent classification (more accurate than keyword matching)
- ✅ Handles metadata queries efficiently without LLM generation
- ✅ Supports partial name queries using semantic search
- ✅ Clear separation of execution paths
- ✅ Fallback to rule-based routing for robustness

### Testing

Run the system:
```bash
uv run python src/main_rag.py
```

Example queries:
- "What are the sample preparation methods?" → SEMANTIC_QUERY
- "What is the ELN ID for cat eats fish?" → METADATA_QUERY
- "Compare project metrics" → DATA_ANALYSIS

---

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
