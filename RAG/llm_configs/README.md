# LLM Configurations

This directory contains all LLM configurations for different components of the RAG system.

## Structure

```
llm_configs/
├── router_config.yaml          # Router LLM (intent classification)
├── rag_config.yaml             # RAG LLM (semantic queries)
├── metadata_config.yaml        # Metadata queries (no LLM)
├── data_analysis_config.yaml   # Data analysis (placeholder)
└── README.md                   # This file
```

## Configuration Files

### 1. router_config.yaml
**Purpose:** Intent classification to route queries to appropriate paths

**LLM Settings:**
- Model: glm-4.5-flash
- Temperature: 0.0 (deterministic)
- Max Tokens: 200 (quick classification)

**Contains:**
- System prompt for classification
- Route definitions (METADATA_QUERY, SEMANTIC_QUERY, DATA_ANALYSIS)
- Fallback keywords for rule-based routing

### 2. rag_config.yaml
**Purpose:** Generate answers from document content (semantic queries)

**LLM Settings:**
- Model: glm-4.5-flash
- Temperature: 0.7 (balanced creativity)
- Max Tokens: 2000 (detailed answers)

**Contains:**
- System prompt for research assistant
- User message template
- Error messages

### 3. metadata_config.yaml
**Purpose:** Configure metadata query responses (no LLM generation)

**Contains:**
- Response templates
- Project item formatting
- No results message

**Note:** Metadata queries use vector search only, no LLM generation

### 4. data_analysis_config.yaml
**Purpose:** Data analysis and visualization (placeholder)

**LLM Settings:**
- Model: glm-4.5-flash
- Temperature: 0.3 (factual)
- Max Tokens: 1500

**Contains:**
- System prompt (placeholder)
- Future: Chart generation, metrics comparison

## Adding New LLM Components

When adding a new LLM-powered feature:

1. Create `{feature}_config.yaml` in this directory
2. Define model settings and prompts
3. Load in code using `load_llm_config()`
4. Update this README

**Example:** Adding a summarization LLM

```yaml
# llm_configs/summarization_config.yaml
model: "glm-4.5-flash"
temperature: 0.5
max_tokens: 1000

system_prompt: |
  You are a summarization assistant...
```

```python
# In code
from src.utils import load_llm_config

summary_config = load_llm_config("summarization_config.yaml")
llm_client = GLMClient(
    model=summary_config["model"],
    temperature=summary_config["temperature"]
)
```

## Usage in Code

### Loading Configs

```python
from src.utils import load_llm_config

# Load router config
router_config = load_llm_config("router_config.yaml")
router_prompt = router_config["system_prompt"]
router_temp = router_config["temperature"]

# Load RAG config
rag_config = load_llm_config("rag_config.yaml")
rag_prompt = rag_config["system_prompt"]
```

### Full Example

```python
# Router
router_config = load_llm_config("router_config.yaml")
router_llm = GLMClient(
    model=router_config["model"],
    temperature=router_config["temperature"]
)

# RAG
rag_config = load_llm_config("rag_config.yaml")
rag_llm = GLMClient(
    model=rag_config["model"],
    temperature=rag_config["temperature"]
)
```

## Configuration Parameters

### Common Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| model | GLM model name | "glm-4.5-flash" |
| temperature | Sampling temperature (0-1) | 0.7 |
| max_tokens | Maximum response tokens | 2000 |
| system_prompt | System instructions | "You are..." |

### Temperature Guide

| Value | Use Case | Example |
|-------|----------|---------|
| 0.0 | Deterministic | Router classification |
| 0.3 | Factual | Data analysis |
| 0.7 | Balanced | RAG answers |
| 1.0+ | Creative | Story generation |

### Max Tokens Guide

| Value | Use Case |
|-------|----------|
| 50-200 | Short responses, classifications |
| 500-1000 | Moderate answers |
| 1500-2000 | Detailed explanations |
| 3000+ | Comprehensive reports |

## Best Practices

### 1. Separation of Concerns
- Each LLM component has its own config file
- Easy to tune without affecting others
- Clear documentation of purpose

### 2. Prompt Engineering
- Be specific about the task
- Include examples if needed
- Define expected output format
- Set clear boundaries

### 3. Version Control
- Commit config changes with code changes
- Document major prompt updates
- Test prompt changes thoroughly

### 4. Testing
```python
# Test a config change
config = load_llm_config("router_config.yaml")
print(f"Temperature: {config['temperature']}")
print(f"Prompt length: {len(config['system_prompt'])}")
```

## Troubleshooting

### Config not loading
```python
# Check if file exists
from pathlib import Path
config_path = Path("llm_configs/router_config.yaml")
print(f"Exists: {config_path.exists()}")

# Check YAML syntax
import yaml
with open(config_path) as f:
    config = yaml.safe_load(f)
```

### Wrong responses
1. Check temperature setting
2. Review system prompt clarity
3. Verify max_tokens is sufficient
4. Test with different queries

### Performance issues
- Lower max_tokens if responses are slow
- Use temperature=0.0 for faster deterministic responses
- Cache configurations in memory

## Migration from Old Structure

**Old:** All prompts in `prompts/prompts.yaml`
```yaml
router:
  system_prompt: ...
semantic_query:
  system_prompt: ...
```

**New:** Separate files per component
```
llm_configs/
├── router_config.yaml
└── rag_config.yaml
```

**Benefits:**
- ✅ Clearer organization
- ✅ Easier to manage individual components
- ✅ Better for version control
- ✅ Scalable for future LLM tools

## Future LLM Components

Planned additions to this directory:

- `translation_config.yaml` - Multi-language support
- `summarization_config.yaml` - Document summarization
- `qa_generation_config.yaml` - Generate Q&A pairs
- `code_generation_config.yaml` - Generate code from specs
- `image_caption_config.yaml` - Image understanding (if multimodal)

Each new LLM tool gets its own config file for clean separation.
