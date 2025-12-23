# RAG System Usage Guide

## Overview

The RAG system retrieves relevant document chunks and uses GLM-4-Flash to generate answers based on the retrieved context.

## Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Set Environment Variables

```bash
# Set ZHIPU AI API key
export ZHIPU_API_KEY="your_api_key_here"
```

Or create a `.env` file (copy from `.env.example`):
```bash
cp .env.example .env
# Edit .env and add your API key
```

### 3. Preprocess Documents

If you haven't already preprocessed your documents:

```bash
uv run python preprocess/chunk_documents.py
```

This creates `preprocess/chunks.json` with all document chunks.

## Running the RAG System

### Interactive Mode

```bash
uv run python src/main_rag.py
```

Then enter your queries:
```
Enter your query (or 'quit' to exit): What authentication methods are supported?
======================================================================
Response:
======================================================================
Based on the provided context, the authentication module supports three methods:
1. Username/Password authentication...
======================================================================
```

## Configuration

All parameters are in `config.yaml`:

### RAG Parameters

```yaml
rag:
  chunks_file: "preprocess/chunks.json"  # Path to chunks
  top_k: 5                                # Number of chunks to retrieve
  min_similarity: 0.0                     # Minimum similarity threshold
```

**Adjusting top_k:**
- Increase for more context (may include less relevant chunks)
- Decrease for more focused answers (may miss relevant info)
- Default: 5 is a good balance

### API Parameters

```yaml
api:
  model: "glm-4-flash"      # GLM model to use
  timeout: 30               # Request timeout (seconds)
  max_retries: 3            # Retry attempts
```

### Logging

```yaml
logging:
  level: "INFO"             # DEBUG, INFO, WARNING, ERROR
  file: "logs/rag_system.log"
  console: true
```

Logs are saved to `logs/rag_system.log` and shown in console.

### Prompts

Customize system prompt and template:

```yaml
prompts:
  system_prompt: |
    You are a helpful AI assistant...

  rag_template: |
    Context:
    {context}

    Question: {query}
```

## Architecture

### Pipeline Flow

```
User Query
    ↓
1. Chunk Retrieval (TF-IDF similarity)
    ↓
2. Context Formatting
    ↓
3. LLM Generation (GLM-4-Flash)
    ↓
Answer
```

### Components

1. **ChunkRetriever** (`src/retriever.py`)
   - Loads preprocessed chunks
   - Builds TF-IDF index
   - Returns top-k similar chunks

2. **GLMClient** (`src/llm_client.py`)
   - ZHIPU AI API client
   - Generates answers from context

3. **RAGNode** (`src/graph/rag_node_new.py`)
   - Orchestrates retrieval + generation
   - Handles errors and logging

## Logging

Logs follow project standards:

- **INFO**: High-level events (query received, chunks retrieved, answer generated)
- **DEBUG**: Internal details (API calls, chunk scores, context length)
- **HTTP logs**: Suppressed to WARNING level

View logs:
```bash
tail -f logs/rag_system.log
```

## Troubleshooting

### No chunks retrieved

**Problem:** "I couldn't find relevant information..."

**Solutions:**
- Check `preprocess/chunks.json` exists
- Lower `min_similarity` in config.yaml
- Increase `top_k`
- Verify query matches document content

### API errors

**Problem:** "Failed to generate answer"

**Solutions:**
- Check `ZHIPU_API_KEY` is set correctly
- Verify API key is valid
- Check internet connection
- Review logs: `logs/rag_system.log`

### Import errors

**Problem:** `ModuleNotFoundError`

**Solutions:**
```bash
# Reinstall dependencies
uv sync

# Check you're using uv run
uv run python src/main_rag.py
```

## Advanced Usage

### Programmatic Usage

```python
from src.utils import load_config, setup_logging
from src.retriever import ChunkRetriever
from src.llm_client import GLMClient
from src.graph.rag_node_new import RAGNode

# Initialize
config = load_config()
logger = setup_logging(config)

retriever = ChunkRetriever(
    chunks_file="preprocess/chunks.json",
    top_k=5
)

llm_client = GLMClient(
    api_key=os.getenv("ZHIPU_API_KEY"),
    model="glm-4-flash"
)

rag_node = RAGNode(
    retriever=retriever,
    llm_client=llm_client,
    system_prompt="You are a helpful assistant."
)

# Process query
state = {"query": "What is OAuth2?"}
result = rag_node.process(state)
print(result["response"])
```

### Custom Similarity Method

Currently uses TF-IDF. To add embeddings:

1. Update `config.yaml`:
```yaml
rag:
  similarity_method: "embedding"

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
```

2. Implement embedding retriever in `src/retriever.py`

## Performance

- **TF-IDF**: Fast, deterministic, good for keyword matching
- **Retrieval**: ~100ms for 73 chunks
- **LLM Generation**: ~2-5s depending on context size

## Next Steps

- Add embedding-based similarity
- Implement hybrid search (TF-IDF + embeddings)
- Add caching for repeated queries
- Integrate vector database (ChromaDB/FAISS)
