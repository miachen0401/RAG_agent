# Quick Start Guide

Get up and running with the LangGraph RAG Agent in 3 simple steps.

## Prerequisites

- Python 3.9 or higher
- `uv` package manager (recommended) or `pip`

## Setup

### Step 1: Install Dependencies with uv

Using `uv` (recommended - handles venv automatically):
```bash
uv sync
```

This will:
- Create a virtual environment automatically
- Install all dependencies
- Generate `uv.lock` for reproducible builds

### Step 2: Run Examples

Run the example script to see both paths in action:
```bash
uv run python example.py
```

Or start interactive mode:
```bash
uv run python run.py
```

### Alternative: Manual Setup

If you prefer manual setup or using `pip`:

1. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install langgraph typing-extensions
```

3. Run examples:
```bash
python example.py
```

## What's Included

The project comes with:

1. **Sample Documents** (`sample_documents/design_doc.txt`)
   - Example authentication module documentation
   - Used for demonstrating RAG path

2. **Example Script** (`example.py`)
   - Demonstrates both RAG and Analysis paths
   - Shows 4 different query types

3. **Interactive Mode** (`src/main.py`)
   - Enter queries interactively
   - Type 'quit' to exit

## Example Queries

Try these queries in interactive mode:

**RAG Path** (document queries):
```
What authentication methods are supported?
What are the configuration options?
Tell me about session management
```

**Analysis Path** (comparison/metrics):
```
Compare project A vs project B
Which model performs better?
Analyze the metrics
```

## Next Steps

1. **Add Your Documents**
   - Place `.txt` files in `sample_documents/` directory
   - Or update `LOCAL_DOC_PATH` in `src/graph/rag_node.py`

2. **Customize Routing**
   - Edit keywords in `src/graph/router.py`
   - Or implement LLM-based routing (when API available)

3. **Connect Database** (for Analysis path)
   - Update `src/graph/analysis_node.py`
   - Pass database connection to `AnalysisNode`

4. **Add Vector Database** (for RAG path)
   - Install: `uv pip install -e ".[future]"`
   - Implement in `src/graph/rag_node.py`

## Troubleshooting

### Import Errors
Make sure you installed the package:
```bash
uv pip install -e .
```

### No Documents Found
Check that `sample_documents/` directory exists and contains `.txt` files.

### Module Not Found
Make sure you're in the project root directory and virtual environment is activated.

## Project Structure

```
.
├── src/
│   ├── graph/
│   │   ├── router.py         # Intent routing
│   │   ├── rag_node.py       # RAG execution path
│   │   └── analysis_node.py  # Analysis execution path
│   └── main.py               # Entry point
├── sample_documents/         # Example documents
├── example.py                # Quick demo script
├── README.md                 # Full documentation
└── QUICKSTART.md            # This file
```

## Need Help?

- See `README.md` for detailed documentation
- See `Instruction.md` for design philosophy
- Check inline `TODO` comments for extension points

Happy coding!
