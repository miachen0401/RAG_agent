# Configuration Structure

This document explains how LLM configurations are separated between different components.

## Two Configuration Files

### 1. `config.yaml` - System Configuration
**Purpose:** Infrastructure and RAG system settings

```yaml
api:  # For RAG LLM (semantic queries)
  model: "glm-4.5-flash"
  timeout: 30
  max_retries: 3
  max_tokens: 2000  # For detailed RAG responses

rag:  # Vector retrieval settings
  top_k: 5
  collection_name: "document_chunks"
  chroma_db_path: "preprocess/output/chroma_db"

embedding:  # Embedding generation
  model: "embedding-3"
  batch_size: 16
  max_length: 8192
```

### 2. `prompts/prompts.yaml` - LLM Prompts & Parameters
**Purpose:** Prompts and their associated LLM parameters

```yaml
router:  # Router LLM settings
  system_prompt: |
    [Intent classification instructions...]
  temperature: 0.0    # Deterministic
  max_tokens: 200     # Allow reasoning

semantic_query:  # RAG LLM prompts
  system_prompt: |
    [Research assistant instructions...]
  temperature: 0.7    # Balanced
  max_tokens: 2000    # Detailed answers
```

## Configuration Separation

### Router LLM
- **Config Source:** `prompts/prompts.yaml` → `router` section
- **Model:** glm-4.5-flash
- **Temperature:** 0.0 (deterministic classification)
- **Max Tokens:** 200 (allow for reasoning)
- **Purpose:** Intent classification only

### RAG LLM (Semantic Query)
- **Config Source:**
  - Model/timeout/retries: `config.yaml` → `api` section
  - Prompts/temperature: `prompts/prompts.yaml` → `semantic_query` section
- **Model:** glm-4.5-flash (from config.yaml)
- **Temperature:** 0.7 (from prompts.yaml)
- **Max Tokens:** 2000 (from config.yaml)
- **Purpose:** Answer generation from document context

## Why Separate?

**Different Responsibilities:**
- **Router:** Quick classification (low max_tokens, temperature=0)
- **RAG:** Detailed answers (high max_tokens, temperature=0.7)

**Easy Tuning:**
- Change router behavior: Edit `prompts/prompts.yaml`
- Change RAG behavior: Edit both files
- No code changes needed

## Loading Flow

```python
# In main_rag.py

# Load system config
config = load_config("config.yaml")
api_config = config["api"]

# Load prompts config
prompts = load_prompts("prompts/prompts.yaml")
router_config = prompts["router"]
semantic_config = prompts["semantic_query"]

# Create Router LLM
llm_client_router = GLMClient(
    model="glm-4.5-flash",
    temperature=0.0,  # Router is deterministic
    timeout=api_config["timeout"],
    max_retries=api_config["max_retries"]
)

# Router function loads its own config
route = route_query(query, llm_client_router)
# Inside route_query:
#   - Loads prompts.yaml
#   - Gets router.max_tokens (200)
#   - Gets router.system_prompt
#   - Calls LLM with these settings

# Create RAG LLM
llm_client_rag = GLMClient(
    model=api_config["model"],
    temperature=0.7,  # From GLMClient default
    timeout=api_config["timeout"],
    max_retries=api_config["max_retries"]
)

# RAG node uses semantic_query prompts
rag_node = RAGNode(
    retriever=retriever,
    llm_client=llm_client_rag,
    system_prompt=semantic_config["system_prompt"]
)
```

## Parameter Priority

### Router
1. **Prompt:** `prompts.yaml` → `router.system_prompt`
2. **Temperature:** `prompts.yaml` → `router.temperature` (0.0)
3. **Max Tokens:** `prompts.yaml` → `router.max_tokens` (200)
4. **Model:** Hardcoded in router.py (`glm-4.5-flash`)
5. **Timeout/Retries:** `config.yaml` → `api` section

### RAG (Semantic Query)
1. **Prompt:** `prompts.yaml` → `semantic_query.system_prompt`
2. **Temperature:** `prompts.yaml` → `semantic_query.temperature` (0.7)
3. **Max Tokens:** `config.yaml` → `api.max_tokens` (2000)
4. **Model:** `config.yaml` → `api.model`
5. **Timeout/Retries:** `config.yaml` → `api` section

## Quick Reference

| Parameter | Router Source | RAG Source |
|-----------|--------------|------------|
| Prompt | prompts.yaml | prompts.yaml |
| Temperature | prompts.yaml (0.0) | prompts.yaml (0.7) |
| Max Tokens | prompts.yaml (200) | config.yaml (2000) |
| Model | router.py code | config.yaml |
| Timeout | config.yaml | config.yaml |
| Retries | config.yaml | config.yaml |

## Editing Configurations

### To change router behavior:
```bash
# Edit router prompt and parameters
nano prompts/prompts.yaml
# Modify router.system_prompt, router.temperature, router.max_tokens
```

### To change RAG answers:
```bash
# Edit RAG prompt
nano prompts/prompts.yaml
# Modify semantic_query.system_prompt

# Edit RAG model parameters
nano config.yaml
# Modify api.max_tokens, api.model
```

### No restart needed:
Both files are loaded at runtime - changes take effect on next query (may need to restart the interactive session).
