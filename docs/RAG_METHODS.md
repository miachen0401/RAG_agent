# RAG Retrieval Methods

## Current Implementation: TF-IDF

### What is TF-IDF?

**TF-IDF (Term Frequency-Inverse Document Frequency)** is a traditional NLP technique that measures word importance:

- **Term Frequency (TF)**: How often a word appears in a document
- **Inverse Document Frequency (IDF)**: How rare a word is across all documents
- **TF-IDF Score**: TF × IDF = importance of word in document

### Implementation

- **Package**: `scikit-learn`
- **Class**: `TfidfVectorizer`
- **Similarity**: Cosine similarity between TF-IDF vectors
- **Location**: `src/retriever.py:47`

### Example

```
Query: "OAuth2 authentication"
Documents:
  Doc 1: "OAuth2 is a protocol for authorization..."  → High score
  Doc 2: "Username and password authentication..."     → Low score
```

### Pros & Cons

**Pros:**
- ✅ Fast (no neural network inference)
- ✅ Deterministic (same query = same results)
- ✅ No GPU required
- ✅ Good for keyword/exact matching
- ✅ Works offline (no API calls)

**Cons:**
- ❌ Doesn't understand semantics (meaning)
- ❌ "car" and "automobile" are different words
- ❌ Misses paraphrased content
- ❌ Query must use similar vocabulary as documents

### When to Use

- Documents with technical terms and exact names
- Keyword-based queries
- Fast response time needed
- No GPU available

---

## Alternative: Neural Embeddings (Not Yet Implemented)

### What are Embeddings?

**Neural embeddings** convert text into dense vectors that capture semantic meaning:

```
"car"        → [0.2, 0.8, 0.1, ...]
"automobile" → [0.3, 0.7, 0.2, ...]  ← Similar vector!
```

### Popular Options

#### 1. Sentence Transformers (Local)

**Model**: `all-MiniLM-L6-v2` or `all-mpnet-base-v2`

```yaml
# config.yaml
rag:
  similarity_method: "embedding"

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"  # or "cuda" for GPU
```

**Pros:**
- ✅ Understands semantic similarity
- ✅ Runs locally (no API calls)
- ✅ Free

**Cons:**
- ❌ Slower than TF-IDF
- ❌ Requires ~100MB model download
- ❌ GPU recommended for speed

#### 2. OpenAI Embeddings (API)

**Model**: `text-embedding-3-small` or `text-embedding-ada-002`

**Pros:**
- ✅ Very high quality
- ✅ No local computation

**Cons:**
- ❌ Costs money per API call
- ❌ Requires internet
- ❌ Latency from API calls

#### 3. ZHIPU AI Embeddings (API)

**Model**: Could use ZHIPU's embedding API

**Pros:**
- ✅ Same provider as LLM
- ✅ Chinese language optimized

**Cons:**
- ❌ API costs
- ❌ Requires internet

---

## Hybrid Search (Future)

Combine TF-IDF + embeddings for best results:

```python
# Keyword match (TF-IDF)
keyword_score = 0.6

# Semantic match (embeddings)
semantic_score = 0.9

# Final score
final_score = 0.3 * keyword_score + 0.7 * semantic_score
```

**Benefits:**
- Catches exact keyword matches (TF-IDF)
- Understands semantic meaning (embeddings)
- More robust retrieval

---

## Current System Architecture

```
Query: "OAuth2 providers"
    ↓
TfidfVectorizer.transform(query)
    ↓
cosine_similarity(query_vector, chunk_vectors)
    ↓
Top 5 chunks (sorted by score)
    ↓
Format as context
    ↓
GLM-4-Flash generates answer
```

---

## How to Switch Methods (Future)

### Add Embedding Support

1. **Install package:**
```bash
uv pip install sentence-transformers
```

2. **Update config.yaml:**
```yaml
rag:
  similarity_method: "embedding"

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
```

3. **Update `src/retriever.py`:**
```python
from sentence_transformers import SentenceTransformer

class EmbeddingRetriever:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query):
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)
        # ... rest of logic
```

---

## Performance Comparison

| Method | Speed | Quality | Cost | Offline |
|--------|-------|---------|------|---------|
| TF-IDF | ⚡⚡⚡ | ⭐⭐ | Free | ✅ |
| Sentence-Transformers | ⚡⚡ | ⭐⭐⭐⭐ | Free | ✅ |
| OpenAI Embeddings | ⚡ | ⭐⭐⭐⭐⭐ | $$$ | ❌ |
| Hybrid | ⚡⚡ | ⭐⭐⭐⭐⭐ | Free | ✅ |

---

## Recommendation

**Current setup (TF-IDF) is good for:**
- Getting started quickly
- Technical documentation with specific terms
- Fast prototyping

**Upgrade to embeddings when:**
- Users ask questions in different ways than documents
- Need semantic understanding
- Have GPU or can tolerate slower retrieval
- Quality > Speed

**Use API embeddings when:**
- Budget available
- Highest quality needed
- Don't want to manage models
