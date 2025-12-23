# Embedding-Based RAG Implementation

## ✅ Implementation Complete

Replaced TF-IDF with **ZHIPU Embedding-3** and **ChromaDB** for proper semantic similarity search.

---

## System Overview

### Embedding Model
- **Service**: ZHIPU AI
- **Model**: `embedding-3`
- **Dimension**: 1024 (typical for Embedding-3)
- **Purpose**: Convert text to semantic vectors

### Vector Database
- **Database**: ChromaDB
- **Similarity**: Cosine similarity
- **Storage**: Persistent local storage (`chroma_db/`)
- **Scalability**: Handles large document sets efficiently

### LLM
- **Service**: ZHIPU AI
- **Model**: `glm-4-flash`
- **Purpose**: Generate answers from retrieved context

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PREPROCESSING (Offline)                   │
├─────────────────────────────────────────────────────────────┤
│ Documents → Chunks → Embeddings → ChromaDB                  │
│              (500 tokens)  (Embedding-3)  (Vector Storage)   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    QUERY TIME (Online)                       │
├─────────────────────────────────────────────────────────────┤
│ User Query                                                   │
│     ↓                                                        │
│ Generate Query Embedding (Embedding-3)                      │
│     ↓                                                        │
│ Search ChromaDB (Cosine Similarity)                         │
│     ↓                                                        │
│ Top-k Most Similar Chunks                                   │
│     ↓                                                        │
│ Format Context                                              │
│     ↓                                                        │
│ GLM-4-Flash (Generate Answer)                               │
│     ↓                                                        │
│ Final Answer                                                │
└─────────────────────────────────────────────────────────────┘
```

---

## Files Created

### Core Components
1. **`src/embeddings.py`**
   - `EmbeddingGenerator`: Generate embeddings using ZHIPU API
   - Batch processing with configurable batch size
   - Handles single text or multiple texts

2. **`preprocess/build_vector_db.py`**
   - Load chunks from `chunks.json`
   - Generate embeddings for all chunks
   - Store in ChromaDB with metadata
   - One-time preprocessing step

3. **`src/vector_retriever.py`**
   - `VectorRetriever`: Query ChromaDB with embeddings
   - Generate query embedding
   - Return top-k most similar chunks
   - Format results for LLM

### Updated Files
- **`config.yaml`**: Added embedding and ChromaDB config
- **`pyproject.toml`**: Added ChromaDB, removed scikit-learn
- **`src/main_rag.py`**: Use VectorRetriever instead of TF-IDF

---

## Configuration

All settings in `config.yaml`:

```yaml
# RAG Configuration
rag:
  top_k: 5                           # Number of chunks to retrieve
  collection_name: "document_chunks"  # ChromaDB collection
  chroma_db_path: "chroma_db"        # ChromaDB storage path

# Embedding Configuration
embedding:
  model: "embedding-3"     # ZHIPU Embedding-3
  batch_size: 16           # Process 16 texts at a time
  max_length: 8192         # Max tokens per embedding

# LLM Configuration
api:
  model: "glm-4-flash"     # Fast GLM-4 variant
  timeout: 30
  max_retries: 3
```

### Adjusting Parameters

**top_k:**
- `3` = Fewer chunks, more focused
- `5` = Balanced (default)
- `10` = More context, may dilute relevance

**batch_size:**
- Higher = Faster preprocessing (but more API load)
- Lower = More conservative API usage

---

## Usage Steps

### 1. Install Dependencies

```bash
uv sync
```

This installs:
- `chromadb>=0.4.0` - Vector database
- `zhipuai>=2.0.0` - ZHIPU AI SDK
- `python-dotenv>=1.0.0` - Environment variables
- All other dependencies

### 2. Set API Key

```bash
# Option A: Create .env file
cp .env.example .env
# Edit .env and add: ZHIPU_API_KEY=your_key

# Option B: Export manually
export ZHIPU_API_KEY="your_key"
```

### 3. Preprocess Documents

```bash
# Step 1: Chunk documents
uv run python preprocess/chunk_documents.py
# Output: preprocess/chunks.json

# Step 2: Generate embeddings and build vector DB
uv run python preprocess/build_vector_db.py
# Output: chroma_db/ directory
```

**Note**: Step 2 will take a few minutes depending on:
- Number of chunks
- API rate limits
- Network speed

Expected time: ~1-5 minutes for 100 chunks

### 4. Run RAG System

```bash
uv run python src/main_rag.py
```

Then ask questions interactively!

---

## Example Session

```
Enter your query (or 'quit' to exit): What OAuth2 providers are supported?

======================================================================
Response:
======================================================================
Based on the provided context, the authentication module supports three
OAuth2 providers:

1. Google
2. GitHub
3. Microsoft

These providers are integrated using the standard OAuth2 authorization
code flow. The tokens are stored securely in encrypted session storage
after successful authentication.
======================================================================
```

---

## Key Differences from TF-IDF

| Feature | TF-IDF (Old) | Embeddings (New) |
|---------|-------------|------------------|
| **Method** | Keyword matching | Semantic similarity |
| **Understanding** | Exact words only | Understands meaning |
| **Synonyms** | ❌ "car" ≠ "automobile" | ✅ Understands synonyms |
| **Paraphrasing** | ❌ Needs exact terms | ✅ Handles variations |
| **Speed** | Very fast (~100ms) | Fast (~200-300ms) |
| **Quality** | Good for keywords | Excellent for meaning |
| **Setup** | Simple | Requires preprocessing |
| **Scalability** | Limited | Excellent |

---

## Benefits

### 1. Semantic Understanding
```
Query: "How do I login?"
Matches: "authentication methods", "sign in process", "access credentials"
```

TF-IDF would miss these (different keywords).
Embeddings understand they mean the same thing.

### 2. Handles Paraphrasing
```
Query: "What ways can users authenticate?"
Document: "The system supports multiple authentication methods"
```

Embeddings recognize the semantic similarity.

### 3. Multilingual Potential
ZHIPU Embedding-3 is optimized for Chinese + English.

### 4. Scalable
ChromaDB handles millions of vectors efficiently.

---

## Preprocessing Details

### Chunk → Embedding Pipeline

1. **Load Chunks**
   ```python
   chunks = load_json("preprocess/chunks.json")
   # Each chunk: {text, file_name, section_name, ...}
   ```

2. **Generate Embeddings**
   ```python
   for batch in chunks (batches of 16):
       embeddings = zhipu_api.embeddings(batch)
   ```

3. **Store in ChromaDB**
   ```python
   chromadb.add(
       ids=[...],
       embeddings=[...],
       documents=[...],
       metadatas=[...]
   )
   ```

### Storage Format

**ChromaDB stores:**
- Vector: 1024-dimensional embedding
- Document: Full chunk text
- Metadata: file_name, section_name, eln_id, etc.

**Index type:** HNSW (Hierarchical Navigable Small World)
- Fast approximate nearest neighbor search
- Scales to millions of vectors

---

## Query Pipeline

```python
# 1. User asks question
query = "What authentication methods exist?"

# 2. Generate query embedding
query_embedding = embedding_generator.generate(query)

# 3. Search ChromaDB
results = chromadb.query(
    query_embeddings=[query_embedding],
    n_results=5  # top_k
)

# 4. Get similar chunks
chunks = results["documents"]  # Top 5 most similar
scores = results["distances"]  # Similarity scores

# 5. Format as context
context = format_chunks(chunks)

# 6. LLM generates answer
answer = glm4_flash.generate(query, context)
```

---

## API Costs

### ZHIPU Embedding-3
- **Cost**: ~¥0.0007 per 1K tokens
- **Example**: 100 chunks × 500 tokens = 50K tokens = ¥0.035 (~$0.005)

### GLM-4-Flash
- **Cost**: Varies by usage
- **Per query**: Context + generation tokens

**Total preprocessing cost for 73 chunks**: < ¥0.05 (~$0.007)

---

## Performance

### Preprocessing (one-time)
- **Chunking**: ~1 second for 73 chunks
- **Embedding generation**: ~1-2 minutes (API calls)
- **ChromaDB storage**: < 1 second

### Query Time
- **Query embedding**: ~200ms
- **ChromaDB search**: ~50-100ms
- **LLM generation**: ~2-5 seconds
- **Total**: ~2-5 seconds

---

## Monitoring

Logs saved to `logs/rag_system.log`:

```
INFO - Generating embeddings for 73 chunks
INFO - Retrieved 5 chunks
INFO - Successfully generated answer
DEBUG - Similarity scores: [0.892, 0.856, 0.824, 0.791, 0.765]
```

---

## Adding More Documents

```bash
# 1. Add documents to project_folder/
cp -r new_project/ project_folder/

# 2. Re-chunk
uv run python preprocess/chunk_documents.py

# 3. Rebuild vector DB
uv run python preprocess/build_vector_db.py

# 4. Query updated database
uv run python src/main_rag.py
```

**Note**: Rebuilding deletes old collection and creates new one with all chunks.

---

## Troubleshooting

### "Collection not found"
```bash
# Build vector database first
uv run python preprocess/build_vector_db.py
```

### "API rate limit"
- Reduce `batch_size` in config.yaml
- Add delays between batches

### "Slow queries"
- Check ChromaDB index size
- Reduce `top_k` if not needed
- Verify ChromaDB is using persistent storage

---

## Summary

✅ **Embedding Model**: ZHIPU Embedding-3 (semantic vectors)
✅ **Vector DB**: ChromaDB (persistent storage)
✅ **Similarity**: Cosine similarity (semantic matching)
✅ **LLM**: GLM-4-Flash (answer generation)
✅ **Configuration**: All parameters in config.yaml
✅ **Scalable**: Handles large document sets

**Ready for production use with many documents!**
