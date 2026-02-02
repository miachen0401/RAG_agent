# Document Preprocessing

This directory handles document preprocessing for the RAG system.

## Directory Structure

```
preprocess/
├── scripts/
│   ├── __init__.py
│   └── build_rag_index.py    # Main script: chunk + embed + store
├── utils/
│   ├── __init__.py
│   ├── chunking.py            # Token-based chunking utilities
│   └── document_loader.py     # Document loading and section extraction
├── output/
│   ├── chunks.json            # Generated chunks (gitignored)
│   ├── chunk_stats.json       # Statistics (gitignored)
│   ├── chroma_db/             # Vector database (gitignored)
│   └── .gitignore
└── README.md                  # This file
```

## Quick Start

### Single Command

Build complete RAG index (chunks + embeddings + vector DB):

```bash
uv run python preprocess/scripts/build_rag_index.py
```

This will:
1. **Chunk documents** from `project_folder/`
2. **Generate embeddings** using ZHIPU Embedding-3
3. **Build ChromaDB** vector database

### Prerequisites

1. Install dependencies:
```bash
uv sync
```

2. Set API key:
```bash
cp .env.example .env
# Edit .env and add your ZHIPU_API_KEY
```

3. Add documents to `project_folder/`:
```
project_folder/
├── project1/
│   └── document.docx
├── project2/
│   └── report.docx
└── ...
```

## What It Does

### Step 1: Document Chunking

- Loads documents from `project_folder/`
- Extracts sections based on headings
- Creates 500-token chunks with 80-token overlap
- Saves to `preprocess/output/chunks.json`

### Step 2: Embedding Generation

- Generates embeddings for each chunk
- Uses ZHIPU Embedding-3 API
- Processes in batches of 16

### Step 3: Vector Database

- Stores chunks and embeddings in ChromaDB
- Uses cosine similarity
- Persists to `preprocess/output/chroma_db/`

## Output Files

### `output/chunks.json`

All chunks with metadata:
```json
[
  {
    "chunk_id": 0,
    "global_chunk_id": 0,
    "text": "Content here...",
    "token_count": 234,
    "file_name": "project1",
    "section_name": "Introduction",
    "eln_id": "ELN0010425"
  }
]
```

### `output/chunk_stats.json`

Processing statistics:
```json
{
  "total_documents": 3,
  "total_chunks": 73,
  "total_tokens": 2273,
  "processing_date": "2025-12-22T..."
}
```

### `output/chroma_db/`

ChromaDB vector database (binary files)

## Configuration

All settings in `config.yaml`:

```yaml
# Chunking
chunking:
  chunk_size: 500    # Tokens per chunk
  overlap: 80        # Overlap tokens

# Embeddings
embedding:
  model: "embedding-3"
  batch_size: 16

# Vector DB
rag:
  collection_name: "document_chunks"
  chroma_db_path: "preprocess/output/chroma_db"
```

## Adding More Documents

```bash
# 1. Add documents to project_folder/
cp -r new_project/ project_folder/

# 2. Rebuild index (overwrites existing)
uv run python preprocess/scripts/build_rag_index.py
```

## Modules

### `utils/chunking.py`

Token-based text chunking:
- `TokenChunker`: Main chunker class
- `create_chunker()`: Factory function
- Uses tiktoken (cl100k_base)

### `utils/document_loader.py`

Document loading:
- `DocumentLoader`: Load .docx and .txt files
- `find_documents_in_folders()`: Discover documents
- Section extraction based on headings
- ELN ID detection

### `scripts/build_rag_index.py`

Main preprocessing pipeline:
- Combines all steps
- Progress logging
- Error handling
- Statistics generation

## Troubleshooting

### "No documents found"

Check `project_folder/` structure:
```bash
ls -R project_folder/
```

Expected: Each subfolder should have a .docx or .txt file.

### "ZHIPU_API_KEY not set"

```bash
# Check .env file
cat .env

# Should contain:
ZHIPU_API_KEY=your_actual_key_here
```

### "ModuleNotFoundError"

```bash
# Reinstall dependencies
uv sync
```

### Slow processing

Reduce batch size in `config.yaml`:
```yaml
embedding:
  batch_size: 8  # Reduce from 16
```

## Performance

For 73 chunks (typical project):
- **Chunking**: ~1-2 seconds
- **Embedding generation**: ~2-3 minutes (API calls)
- **ChromaDB storage**: ~1 second
- **Total**: ~3-5 minutes

## API Costs

ZHIPU Embedding-3: ~¥0.0007 per 1K tokens

Example: 100 chunks × 500 tokens = 50K tokens = ¥0.035 (~$0.005)

## Logging

Logs saved to `logs/rag_system.log`:

```
INFO - Found 3 document(s)
INFO - Processing: project1
INFO - Created 25 chunk(s)
INFO - Generating embeddings for 73 chunks
INFO - Generated 73 embeddings
INFO - Added batch 1/1
INFO - RAG Index Build Complete!
```

## Next Steps

After building the index:

```bash
# Run RAG system
uv run python src/main_rag.py
```

The system will automatically load from `preprocess/output/chroma_db/`.
