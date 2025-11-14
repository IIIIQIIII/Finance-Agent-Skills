---
name: qwen-embedding
description: Text vectorization and semantic analysis using Alibaba Cloud DashScope's text-embedding-v4 model. Provides 6 major capabilities - semantic search, recommendation systems, text clustering, zero-shot classification, anomaly detection, and advanced embedding features (sparse/dense vectors, hybrid search).
---

# Qwen Text Embedding Skill

Comprehensive text vectorization and semantic analysis toolkit using Alibaba Cloud DashScope's text-embedding-v4 model.

## What This Skill Does

This skill provides 6 major text embedding capabilities:

1. **Semantic Search** - Find semantically similar content
2. **Recommendation Systems** - Build user preference-based recommendations
3. **Text Clustering** - Automatically group similar texts
4. **Zero-Shot Classification** - Classify text without training data
5. **Anomaly Detection** - Detect outliers in text data
6. **Advanced Features** - Sparse vectors, hybrid search, task instructions

## Prerequisites

### 1. Install Dependencies

```bash
# Create project directory
mkdir qwen-embedding-project
cd qwen-embedding-project

# Create virtual environment with uv
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
uv pip install dashscope python-dotenv numpy scikit-learn
```

### 2. Configure API Key

Create a `.env` file in your project directory:

```bash
# .env file
DASHSCOPE_API_KEY=your_api_key_here

# Optional: Use Singapore region
# DASHSCOPE_BASE_URL=https://dashscope-intl.aliyuncs.com/api/v1
```

**Security Note**: Never commit `.env` files to version control. Add to `.gitignore`:
```
.env
.env.local
```

### 3. Get API Key

1. Sign up at [Alibaba Cloud DashScope](https://dashscope.console.aliyun.com/)
2. Navigate to API Keys section
3. Create a new API key
4. Copy the key to your `.env` file

## Core Capabilities

### 1. Semantic Search

Find documents most similar to a query using semantic understanding.

**Use Cases**:
- Document retrieval
- FAQ matching
- Content discovery
- Knowledge base search

**Basic Example**:

```python
import os
from http import HTTPStatus
import dashscope
import numpy as np
from dotenv import load_dotenv

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_search(query, documents, top_k=3):
    # Get query embedding
    query_resp = dashscope.TextEmbedding.call(
        model="text-embedding-v4",
        input=query,
        dimension=1024
    )
    query_emb = query_resp.output['embeddings'][0]['embedding']

    # Get document embeddings
    doc_resp = dashscope.TextEmbedding.call(
        model="text-embedding-v4",
        input=documents,
        dimension=1024
    )

    # Calculate similarities
    similarities = []
    for i, doc_emb in enumerate(doc_resp.output['embeddings']):
        sim = cosine_similarity(query_emb, doc_emb['embedding'])
        similarities.append((documents[i], sim))

    # Sort and return top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Example usage
documents = [
    "人工智能是计算机科学的一个分支",
    "机器学习是实现人工智能的重要方法",
    "深度学习是机器学习的一个子领域"
]

results = semantic_search("什么是AI？", documents)
for doc, score in results:
    print(f"[{score:.4f}] {doc}")
```

### 2. Recommendation System

Build personalized recommendations based on user preferences.

**Use Cases**:
- Movie/content recommendations
- Product recommendations
- Article suggestions
- Personalized feeds

**Example**:

```python
def build_user_profile(user_history):
    """Generate user preference vector from history"""
    resp = dashscope.TextEmbedding.call(
        model="text-embedding-v4",
        input=user_history,
        dimension=1024
    )
    embeddings = [emb['embedding'] for emb in resp.output['embeddings']]
    return np.mean(embeddings, axis=0)  # Average vector

def recommend(user_profile, items, top_k=5):
    """Recommend items based on user profile"""
    items_resp = dashscope.TextEmbedding.call(
        model="text-embedding-v4",
        input=items,
        dimension=1024
    )

    recommendations = []
    for i, item_emb in enumerate(items_resp.output['embeddings']):
        score = cosine_similarity(user_profile, item_emb['embedding'])
        recommendations.append((items[i], score))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:top_k]

# Example
user_history = ["科幻电影", "动作片", "悬疑片"]
user_profile = build_user_profile(user_history)

items = ["科幻大片", "浪漫爱情", "悬疑推理", "喜剧片"]
recommendations = recommend(user_profile, items, top_k=3)
```

### 3. Text Clustering

Automatically group similar texts together.

**Use Cases**:
- Topic discovery
- Document organization
- Customer feedback categorization
- Content grouping

**Example** (requires scikit-learn):

```python
from sklearn.cluster import KMeans

def cluster_texts(texts, n_clusters=3):
    # Get embeddings
    resp = dashscope.TextEmbedding.call(
        model="text-embedding-v4",
        input=texts,
        dimension=1024
    )
    embeddings = np.array([emb['embedding'] for emb in resp.output['embeddings']])

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(embeddings)

    # Group texts by cluster
    clusters = {i: [] for i in range(n_clusters)}
    for text, label in zip(texts, labels):
        clusters[label].append(text)

    return clusters

# Example
news = [
    "AI技术突破",
    "机器学习进展",
    "NBA总决赛",
    "世界杯足球赛",
    "深度学习研究"
]
clusters = cluster_texts(news, n_clusters=2)
```

### 4. Zero-Shot Classification

Classify text without training data.

**Use Cases**:
- Product categorization
- Sentiment analysis
- Intent recognition
- Content tagging

**Example**:

```python
def classify_text(text, labels):
    # Get embeddings for text and all labels
    all_texts = [text] + labels
    resp = dashscope.TextEmbedding.call(
        model="text-embedding-v4",
        input=all_texts,
        dimension=1024
    )

    embeddings = resp.output['embeddings']
    text_emb = embeddings[0]['embedding']

    # Find best matching label
    best_label = None
    best_score = -1

    for i, label in enumerate(labels):
        label_emb = embeddings[i + 1]['embedding']
        score = cosine_similarity(text_emb, label_emb)
        if score > best_score:
            best_score = score
            best_label = label

    return best_label, best_score

# Example
text = "这件衣服质量很好"
labels = ["数码产品", "服装配饰", "食品饮料", "家居生活"]
category, confidence = classify_text(text, labels)
print(f"分类: {category} (置信度: {confidence:.4f})")
```

### 5. Anomaly Detection

Detect outliers in text data.

**Use Cases**:
- Spam detection
- Fraud detection
- Quality control
- Content moderation

**Example**:

```python
def detect_anomalies(normal_samples, test_samples, threshold=0.6):
    # Get embeddings for normal samples
    normal_resp = dashscope.TextEmbedding.call(
        model="text-embedding-v4",
        input=normal_samples,
        dimension=1024
    )
    normal_embs = [emb['embedding'] for emb in normal_resp.output['embeddings']]
    normal_center = np.mean(normal_embs, axis=0)

    # Test each sample
    test_resp = dashscope.TextEmbedding.call(
        model="text-embedding-v4",
        input=test_samples,
        dimension=1024
    )

    results = []
    for i, test_emb in enumerate(test_resp.output['embeddings']):
        similarity = cosine_similarity(test_emb['embedding'], normal_center)
        is_anomaly = similarity < threshold
        results.append((test_samples[i], is_anomaly, similarity))

    return results

# Example
normal = ["会议很成功", "项目进展顺利", "功能符合需求"]
test = ["功能完成得很好", "asdfghjkl随机文本", "这是垃圾"]
results = detect_anomalies(normal, test)
```

### 6. Advanced Features

#### A. Text Type Parameter (query vs document)

Optimize vectors for search scenarios:

```python
# For search queries
query_emb = dashscope.TextEmbedding.call(
    model="text-embedding-v4",
    input="深度学习",
    text_type="query",  # Optimized for searching
    dimension=1024
)

# For documents being searched
doc_emb = dashscope.TextEmbedding.call(
    model="text-embedding-v4",
    input="深度学习是机器学习的子领域",
    text_type="document",  # Optimized for being matched
    dimension=1024
)
```

#### B. Task Instructions (instruct)

Guide the model for specific tasks:

```python
# Academic paper search
query_emb = dashscope.TextEmbedding.call(
    model="text-embedding-v4",
    input="机器学习相关论文",
    text_type="query",
    instruct="Given a research paper query, retrieve relevant research papers",
    dimension=1024
)
```

#### C. Sparse Vectors (keyword matching)

For exact keyword matching:

```python
sparse_resp = dashscope.TextEmbedding.call(
    model="text-embedding-v4",
    input="iPhone 15 Pro",
    output_type="sparse",  # Sparse vector for keywords
    dimension=1024
)

# Access sparse vector
sparse_emb = sparse_resp.output['embeddings'][0]['sparse_embedding']
indices = sparse_emb['indices']
values = sparse_emb['values']
```

#### D. Hybrid Search (dense + sparse)

Combine semantic understanding with keyword matching:

```python
# Get hybrid embeddings
hybrid_resp = dashscope.TextEmbedding.call(
    model="text-embedding-v4",
    input="Python机器学习",
    output_type="dense&sparse",  # Both vectors
    dimension=1024
)

# Access both vectors
emb = hybrid_resp.output['embeddings'][0]
dense_vec = emb['embedding']
sparse_vec = emb['sparse_embedding']

# Combine scores
dense_score = cosine_similarity(query_dense, doc_dense)
sparse_score = sparse_dot_product(query_sparse, doc_sparse)
hybrid_score = 0.7 * dense_score + 0.3 * sparse_score
```

## Configuration Options

### Vector Dimensions

Choose based on your accuracy vs cost tradeoff:

- **256**: Fast, lower cost, good for simple tasks
- **512**: Balanced performance
- **1024**: High accuracy (default, recommended)
- **2048**: Highest accuracy, higher cost

```python
resp = dashscope.TextEmbedding.call(
    model="text-embedding-v4",
    input="text",
    dimension=1024  # Choose: 256, 512, 1024, or 2048
)
```

### Model Selection

- **text-embedding-v4**: Latest, best performance (recommended)
- **text-embedding-v3**: Previous version

### Temperature (for instruct mode)

Not directly available, but you can control model behavior through prompt engineering in the instruct parameter.

## Best Practices

### 1. API Key Security

```python
# ✅ GOOD: Load from environment
from dotenv import load_dotenv
import os

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# ❌ BAD: Hardcode in code
dashscope.api_key = "sk-abc123..."  # NEVER DO THIS
```

### 2. Batch Processing

Process multiple texts at once to reduce API calls:

```python
# ✅ GOOD: Batch processing
texts = ["text1", "text2", "text3"]
resp = dashscope.TextEmbedding.call(
    model="text-embedding-v4",
    input=texts,  # All at once
    dimension=1024
)

# ❌ LESS EFFICIENT: One at a time
for text in texts:
    resp = dashscope.TextEmbedding.call(
        model="text-embedding-v4",
        input=text,
        dimension=1024
    )
```

### 3. Caching Embeddings

Cache vectors for frequently accessed documents:

```python
import json

# Cache embeddings
cache = {}
for doc in documents:
    emb = get_embedding(doc)
    cache[doc] = emb

# Save cache
with open('embeddings_cache.json', 'w') as f:
    json.dump(cache, f)

# Load cache
with open('embeddings_cache.json', 'r') as f:
    cache = json.load(f)
```

### 4. Error Handling

```python
from http import HTTPStatus

try:
    resp = dashscope.TextEmbedding.call(
        model="text-embedding-v4",
        input=text,
        dimension=1024
    )

    if resp.status_code == HTTPStatus.OK:
        embedding = resp.output['embeddings'][0]['embedding']
    else:
        print(f"Error: {resp.code} - {resp.message}")

except Exception as e:
    print(f"Exception: {e}")
```

### 5. Use Appropriate text_type

```python
# For user queries
query_emb = get_embedding(query, text_type="query")

# For documents in database
doc_emb = get_embedding(document, text_type="document")

# For clustering/classification (all texts same role)
emb = get_embedding(text)  # Default is fine
```

## Performance Optimization

### 1. Choose Right Dimension

- Simple tasks (FAQ matching): 256 or 512
- General purpose: 1024 (default)
- High precision required: 2048

### 2. Use Hybrid Search for Production

Combine semantic and keyword matching:

```python
# 70% semantic + 30% keywords
score = 0.7 * dense_similarity + 0.3 * sparse_score
```

### 3. Pre-compute Document Embeddings

For search systems:
1. Compute document embeddings offline
2. Store in vector database
3. Only compute query embedding at runtime

### 4. Batch API Calls

- Process multiple texts in single API call
- Reduces network overhead
- Lower overall latency

## Troubleshooting

### Issue: "API Key Not Found"

**Solution**:
```bash
# Check .env file exists and contains key
cat .env

# Ensure python-dotenv is installed
uv pip install python-dotenv

# Load environment in code
from dotenv import load_dotenv
load_dotenv()
```

### Issue: "Rate Limit Exceeded"

**Solution**:
- Add delays between requests
- Use batch processing
- Upgrade API plan if needed

```python
import time

for batch in batches:
    process_batch(batch)
    time.sleep(1)  # Rate limiting
```

### Issue: Poor Search Results

**Solutions**:
1. Use `text_type="query"` for queries and `text_type="document"` for documents
2. Try higher dimension (1024 or 2048)
3. Use hybrid search (dense + sparse)
4. Add task-specific `instruct` parameter

### Issue: High API Costs

**Solutions**:
1. Use lower dimensions (256 or 512) if acceptable
2. Cache embeddings for reused texts
3. Batch process multiple texts
4. Use sparse vectors for keyword-heavy tasks

## Example Projects

### Project 1: FAQ Bot

```python
class FAQBot:
    def __init__(self, faq_data):
        self.questions = [item['question'] for item in faq_data]
        self.answers = [item['answer'] for item in faq_data]
        self.question_embeddings = self._embed_questions()

    def _embed_questions(self):
        resp = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=self.questions,
            text_type="document",
            dimension=1024
        )
        return [emb['embedding'] for emb in resp.output['embeddings']]

    def answer(self, user_question):
        # Get query embedding
        query_resp = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=user_question,
            text_type="query",
            dimension=1024
        )
        query_emb = query_resp.output['embeddings'][0]['embedding']

        # Find best match
        best_idx = 0
        best_score = -1
        for i, q_emb in enumerate(self.question_embeddings):
            score = cosine_similarity(query_emb, q_emb)
            if score > best_score:
                best_score = score
                best_idx = i

        return self.answers[best_idx], best_score
```

### Project 2: Content Recommender

```python
class ContentRecommender:
    def __init__(self):
        self.user_profiles = {}

    def add_user_interaction(self, user_id, content):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = []
        self.user_profiles[user_id].append(content)

    def get_recommendations(self, user_id, candidates, top_k=5):
        # Build user profile
        history = self.user_profiles.get(user_id, [])
        if not history:
            return []

        history_resp = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=history,
            dimension=1024
        )
        profile = np.mean([emb['embedding'] for emb in history_resp.output['embeddings']], axis=0)

        # Score candidates
        cand_resp = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=candidates,
            dimension=1024
        )

        scores = []
        for i, cand_emb in enumerate(cand_resp.output['embeddings']):
            score = cosine_similarity(profile, cand_emb['embedding'])
            scores.append((candidates[i], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

## When to Use This Skill

Use Qwen embedding when you need to:

- **Search and retrieval**: Find similar documents, FAQ matching
- **Recommendations**: Content, product, or user recommendations
- **Classification**: Categorize text without training data
- **Clustering**: Discover topics or group similar content
- **Anomaly detection**: Find outliers in text data
- **Semantic analysis**: Understand text meaning beyond keywords

**Best for**:
- Chinese language processing (native support)
- Multilingual applications
- Production systems (high performance, scalable)
- Cost-sensitive projects (competitive pricing)

## API Reference

### Basic Call

```python
dashscope.TextEmbedding.call(
    model="text-embedding-v4",
    input="text or list of texts",
    dimension=1024,  # Optional: 256, 512, 1024, 2048
    text_type="document",  # Optional: "query" or "document"
    output_type="dense",  # Optional: "dense", "sparse", "dense&sparse"
    instruct="task instruction"  # Optional: requires text_type="query"
)
```

### Response Structure

```python
{
    "status_code": 200,
    "request_id": "...",
    "output": {
        "embeddings": [
            {
                "text_index": 0,
                "embedding": [...],  # Dense vector
                "sparse_embedding": {  # If output_type includes sparse
                    "indices": [...],
                    "values": [...]
                }
            }
        ]
    }
}
```

## Pricing

- Charged per token
- Different dimensions may have different pricing
- Batch calls are more cost-effective
- Sparse vectors included at no extra cost
- Check latest pricing at: https://help.aliyun.com/zh/dashscope/

## Resources

- **Documentation**: https://help.aliyun.com/zh/dashscope/
- **API Reference**: https://help.aliyun.com/zh/dashscope/developer-reference/text-embedding-api-details
- **Code Examples**: Available in Qwen project at /Users/jason/Projects/Qwen/

## Support

For issues or questions:
- DashScope Documentation: https://help.aliyun.com/zh/dashscope/
- Alibaba Cloud Support: https://www.alibabacloud.com/support

---

**Last Updated**: 2025-01-14
**Model Version**: text-embedding-v4
**Skill Maintainer**: Internal
