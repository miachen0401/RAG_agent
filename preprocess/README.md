# Document Preprocessing and Chunking

This module handles offline preprocessing of documents to create token-based chunks for RAG search.

## Overview

The preprocessing pipeline:

1. **Discovers documents** in `project_folder/` directory
2. **Loads documents** (supports .txt and .docx formats)
3. **Extracts sections** based on headings and structure
4. **Chunks text** using deterministic token-based splitting
5. **Annotates chunks** with metadata (file_name, eln_id, section_id, section_name)
6. **Saves chunks** to `chunks.json` for later use

## Features

### Token-Based Chunking

- **Chunk Size**: 500 tokens
- **Overlap**: 80 tokens (prevents truncation of identifiers, file paths, numeric values)
- **Deterministic**: Uses tiktoken (cl100k_base encoding) for reproducible results
- **Section-Aware**: Chunking applied within sections, but boundaries don't constrain chunk size

### Metadata Annotation

Each chunk includes:

- `chunk_id`: Chunk number within the section
- `global_chunk_id`: Unique ID across all chunks
- `text`: Chunk content
- `token_count`: Number of tokens in chunk
- `file_name`: Folder name containing the document
- `file_path`: Full path to source document
- `eln_id`: Electronic Lab Notebook ID (if present in document)
- `section_id`: Section number
- `section_name`: Section heading/title
- `start_token_idx`, `end_token_idx`: Token positions for debugging

### Document Structure Support

**Expected folder structure:**
```
project_folder/
├── project1/
│   └── document.docx    # Will be processed with file_name="project1"
├── project2/
│   └── report.docx      # Will be processed with file_name="project2"
└── design_doc.txt       # Will be processed with file_name="project_folder"
```

**Currently supported:**
- Flat structure (documents directly in `project_folder/`)
- Nested structure (one document per subfolder - preferred structure)

## Usage

### Run Preprocessing

```bash
# Run the preprocessing script
uv run python preprocess/chunk_documents.py
```

### Output Files

After running, two files are created:

1. **`preprocess/chunks.json`** - All chunks with full metadata
   ```json
   [
     {
       "chunk_id": 0,
       "global_chunk_id": 0,
       "text": "System Design Document - Authentication Module...",
       "token_count": 7,
       "file_name": "project_folder",
       "file_path": "project_folder/design_doc.txt",
       "eln_id": null,
       "section_id": 0,
       "section_name": "Introduction",
       "start_token_idx": 0,
       "end_token_idx": 7
     },
     ...
   ]
   ```

2. **`preprocess/chunk_stats.json`** - Processing statistics
   ```json
   {
     "total_documents": 1,
     "total_chunks": 7,
     "total_sections": 7,
     "total_tokens": 309,
     "chunk_size": 500,
     "overlap": 80,
     "processing_date": "2025-12-22T20:17:23.965647",
     "documents": [...]
   }
   ```

## Module Structure

```
preprocess/
├── __init__.py
├── chunking.py           # Token-based chunking utilities
├── document_loader.py    # Document loading and section extraction
├── chunk_documents.py    # Main preprocessing script
├── chunks.json          # Output: processed chunks (generated)
├── chunk_stats.json     # Output: statistics (generated)
└── README.md           # This file
```

## Components

### `chunking.py`

**TokenChunker class:**
- `chunk_text(text, metadata)`: Chunk single text with metadata
- `chunk_sections(sections, base_metadata)`: Chunk multiple sections
- `count_tokens(text)`: Count tokens in text

### `document_loader.py`

**DocumentLoader class:**
- `load_document(file_path)`: Auto-detect and load any supported document
- `load_text_file(file_path)`: Load and parse .txt files
- `load_docx_file(file_path)`: Load and parse .docx files
- `extract_eln_id(text)`: Extract ELN ID from document text

**Utility functions:**
- `find_documents_in_folders(root_dir)`: Find all documents in folder structure

### `chunk_documents.py`

**DocumentPreprocessor class:**
- `process_all_documents()`: Main pipeline
- `save_chunks(chunks)`: Save chunks to JSON
- `save_stats(stats)`: Save statistics to JSON
- `run()`: Complete preprocessing workflow

## Section Detection

Sections are automatically detected using multiple heuristics:

### For .txt files:
- Lines ending with `:` (e.g., "Introduction:")
- All-caps lines (e.g., "AUTHENTICATION MODULE")
- Numbered headings (e.g., "1. Overview", "2.1 Details")
- Markdown headers (e.g., "# Introduction")

### For .docx files:
- Built-in Heading styles (Heading 1, Heading 2, etc.)
- Text patterns matching heading conventions
- Tables are extracted as separate sections

## ELN ID Detection

Automatically detects Electronic Lab Notebook IDs in documents:

**Supported patterns:**
- `ELN-12345`
- `ELN_12345`
- `eln-12345`
- `ELN ID: 12345`
- `eln id: 12345`

## Customization

### Adjust Chunk Size

Edit `chunk_documents.py` or pass parameters:

```python
preprocessor = DocumentPreprocessor(
    chunk_size=1000,  # Change to 1000 tokens
    overlap=100       # Change to 100 token overlap
)
```

### Change Input/Output Directories

```python
preprocessor = DocumentPreprocessor(
    documents_dir="path/to/documents",
    output_dir="path/to/output"
)
```

### Programmatic Usage

```python
from preprocess.chunking import create_chunker
from preprocess.document_loader import DocumentLoader

# Create chunker
chunker = create_chunker(chunk_size=500, overlap=80)

# Load document
loader = DocumentLoader()
doc_data = loader.load_document("project_folder/design_doc.txt")

# Chunk sections
metadata = {"file_name": "my_doc", "eln_id": None}
chunks = chunker.chunk_sections(doc_data["sections"], metadata)
```

## Integration with RAG

The generated `chunks.json` file is ready to be:

1. **Loaded into vector database** (future):
   ```python
   import json
   from chromadb import Client

   with open("preprocess/chunks.json") as f:
       chunks = json.load(f)

   # Add to vector database
   for chunk in chunks:
       vector_db.add(
           text=chunk["text"],
           metadata={...}
       )
   ```

2. **Used directly** (current):
   - Load chunks in memory
   - Search using keyword matching
   - Retrieve relevant chunks for RAG

3. **Migrated to online database**:
   - Import chunks into SQL database
   - Add vector embeddings column
   - Enable hybrid search (keyword + semantic)

## Example Output

```
======================================================================
Document Preprocessing and Chunking
======================================================================
Chunk size: 500 tokens
Overlap: 80 tokens

Searching for documents in: project_folder
Found 1 document(s)

Processing: project_folder (project_folder/design_doc.txt)
  - Found 7 section(s)
  - Created 7 chunk(s)

Saved 7 chunks to: preprocess/chunks.json
Saved statistics to: preprocess/chunk_stats.json

======================================================================
Summary
======================================================================
Total documents processed: 1
Total sections extracted: 7
Total chunks created: 7
Total tokens processed: 309
Average tokens per chunk: 44.1
======================================================================
```

## Future Enhancements

- [ ] Add support for PDF files
- [ ] Implement semantic section detection
- [ ] Add automatic language detection
- [ ] Support for code files with syntax-aware chunking
- [ ] Incremental processing (only update changed documents)
- [ ] Direct integration with vector databases
- [ ] Embedding generation during preprocessing
- [ ] Support for .doc files (older Word format)
- [ ] Multi-threading for large document sets

## Dependencies

Required packages (installed via `uv sync`):
- `tiktoken>=0.5.0` - Token counting and encoding
- `python-docx>=1.0.0` - Word document processing

## Troubleshooting

**No documents found:**
- Ensure documents are in `project_folder/` directory
- Check file extensions (.txt or .docx)
- Verify folder structure matches expected pattern (one .docx per subfolder)

**Import errors:**
- Run `uv sync` to install dependencies
- Ensure you're running from project root

**Small chunks:**
- Documents with many small sections will create smaller chunks
- This is expected behavior to maintain section boundaries
- Average chunk size shown in stats

**Missing ELN ID:**
- Ensure ELN ID follows supported patterns
- Check document contains "ELN" or "eln" prefix
- Add custom patterns in `extract_eln_id()` if needed

## Contact

For questions or issues with preprocessing, refer to the main project README.
