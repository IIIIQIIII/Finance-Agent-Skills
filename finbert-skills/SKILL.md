---
name: finbert-skills
description: Perform financial text analysis using FinBERT models including sentiment analysis, ESG classification, and forward-looking statement detection on financial documents and reports.
---

# FinBERT Financial Analysis Skills

Analyze financial texts using specialized FinBERT models for sentiment, ESG topics, and forward-looking statements.

## What This Skill Does

This skill provides financial text analysis capabilities using FinBERT models:

1. **Sentiment Analysis** - Classify financial text as positive, neutral, or negative
2. **ESG Classification** - Categorize text into Environmental, Social, Governance, or Non-ESG themes
3. **Fine-grained ESG Analysis** - Classify into 9 detailed ESG categories
4. **Forward-Looking Statements** - Identify Specific-FLS, Non-specific FLS, or Not-FLS
5. **Chinese Sentiment Analysis** - Analyze sentiment in Chinese financial texts

All models are based on FinBERT, a BERT model pre-trained on 4.9B tokens from:
- Corporate Reports (10-K & 10-Q): 2.5B tokens
- Earnings Call Transcripts: 1.3B tokens
- Analyst Reports: 1.1B tokens

## Prerequisites

Before using this skill, ensure you have:

1. **Python environment** with transformers library:
   ```bash
   pip install transformers>=4.18.0
   ```

2. **PyTorch** installed (required by transformers)

## Available Models

### 1. FinBERT-Tone (Sentiment Analysis)

**Model**: `yiyanghkust/finbert-tone`

**Purpose**: Classify sentiment in financial texts

**Output Labels**:
- LABEL_0: Neutral
- LABEL_1: Positive
- LABEL_2: Negative

**Use Cases**:
- Analyze earnings call transcripts
- Assess management tone in reports
- Track sentiment changes over time
- Compare sentiment across companies

**Example Usage**:
```python
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

sentences = [
    "there is a shortage of capital, and we need extra financing",
    "growth is strong and we have plenty of liquidity",
    "there are doubts about our finances",
    "profits are flat"
]
results = nlp(sentences)
print(results)
```

**Example Output**:
```python
[
    {'label': 'LABEL_2', 'score': 0.95},  # Negative
    {'label': 'LABEL_1', 'score': 0.98},  # Positive
    {'label': 'LABEL_2', 'score': 0.92},  # Negative
    {'label': 'LABEL_0', 'score': 0.87}   # Neutral
]
```

### 2. FinBERT-ESG (4 Categories)

**Model**: `yiyanghkust/finbert-esg`

**Purpose**: Classify text into broad ESG themes

**Output Labels**:
- Environmental
- Social
- Governance
- Non-ESG

**Use Cases**:
- Screen ESG-related content
- Categorize sustainability reports
- Filter ESG vs non-ESG discussions

**Example Usage**:
```python
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg', num_labels=4)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg')
nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer)

text = "Rhonda has been volunteering for several years for a variety of charitable community programs."
result = nlp(text)
print(result)  # [{'label': 'Social', 'score': 0.9906041026115417}]
```

### 3. FinBERT-ESG-9-Categories (Fine-grained ESG)

**Model**: `yiyanghkust/finbert-esg-9-categories`

**Purpose**: Classify text into detailed ESG topics

**Output Labels**:
1. Climate Change
2. Natural Capital
3. Pollution & Waste
4. Human Capital
5. Product Liability
6. Community Relations
7. Corporate Governance
8. Business Ethics & Values
9. Non-ESG

**Training**: Fine-tuned on ~14,000 manually annotated sentences from ESG reports and annual reports

**Use Cases**:
- Detailed ESG topic analysis
- Sustainability report categorization
- ESG risk assessment
- Compliance monitoring

**Example Usage**:
```python
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg-9-categories', num_labels=9)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg-9-categories')
nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer)

text = "For 2002, our total net emissions were approximately 60 million metric tons of CO2 equivalents for all businesses and operations we have financial interests in, based on its equity share in those businesses and operations."
result = nlp(text)
print(result)  # [{'label': 'Climate Change', 'score': 0.9955655932426453}]
```

**Category Descriptions**: [Download PDF](https://www.allenhuang.org/uploads/2/6/5/5/26555246/esg_9-class_descriptions.pdf)

### 4. FinBERT-FLS (Forward-Looking Statements)

**Model**: `yiyanghkust/finbert-fls`

**Purpose**: Identify forward-looking statements in financial texts

**Output Labels**:
- Specific FLS: Concrete, specific forward-looking statements
- Non-specific FLS: Vague or general forward-looking statements
- Not-FLS: Not a forward-looking statement

**Training**: Fine-tuned on 3,500 manually annotated sentences from MD&A sections of Russell 3000 firms

**Use Cases**:
- Extract management predictions
- Identify specific vs vague guidance
- Analyze forward-looking disclosure patterns
- Support investment research

**Example Usage**:
```python
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-fls', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-fls')
nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer)

text = "We expect the age of our fleet to enhance availability and reliability due to reduced downtime for repairs."
result = nlp(text)
print(result)  # [{'label': 'Specific FLS', 'score': 0.77278733253479}]
```

### 5. FinBERT-Tone-Chinese

**Model**: `yiyanghkust/finbert-tone-chinese`

**Purpose**: Sentiment analysis for Chinese financial texts

**Output Labels**:
- LABEL_0: Neutral
- LABEL_1: Positive
- LABEL_2: Negative

**Training**: Fine-tuned on ~8,000 Chinese analyst report sentences

**Performance**:
- Test Accuracy: 0.88
- Test Macro F1: 0.87

**Use Cases**:
- Analyze Chinese financial reports
- Monitor Chinese market sentiment
- Cross-language sentiment comparison

**Example Usage**:
```python
from transformers import TextClassificationPipeline
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast

model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone-chinese', output_attentions=True)
tokenizer = BertTokenizerFast.from_pretrained('yiyanghkust/finbert-tone-chinese')
nlp = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

text = "此外宁德时代上半年实现出口约2GWh，同比增加200%+。"
result = nlp(text)
print(result)
# [[{'label': 'LABEL_0', 'score': 0.0007}, {'label': 'LABEL_1', 'score': 0.9989}, {'label': 'LABEL_2', 'score': 0.0004}]]
```

## Common Workflow Patterns

### Pattern 1: Sentiment Analysis Pipeline

```python
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# Initialize model
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

# Process document
def analyze_document_sentiment(text):
    # Split into sentences
    sentences = text.split('. ')

    # Analyze each sentence
    results = nlp(sentences)

    # Aggregate results
    sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
    for result in results:
        if result['label'] == 'LABEL_1':
            sentiment_counts['positive'] += 1
        elif result['label'] == 'LABEL_0':
            sentiment_counts['neutral'] += 1
        else:
            sentiment_counts['negative'] += 1

    return sentiment_counts, results

# Use it
text = "Your financial document text here..."
counts, details = analyze_document_sentiment(text)
print(f"Sentiment distribution: {counts}")
```

### Pattern 2: ESG Topic Extraction

```python
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# Initialize model
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg-9-categories', num_labels=9)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg-9-categories')
nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer)

# Extract ESG topics
def extract_esg_topics(paragraphs, threshold=0.5):
    topics = {}

    for para in paragraphs:
        result = nlp(para)
        label = result[0]['label']
        score = result[0]['score']

        if score >= threshold and label != 'Non-ESG':
            if label not in topics:
                topics[label] = []
            topics[label].append({
                'text': para,
                'confidence': score
            })

    return topics

# Use it
paragraphs = ["paragraph 1...", "paragraph 2..."]
esg_topics = extract_esg_topics(paragraphs)
for topic, items in esg_topics.items():
    print(f"{topic}: {len(items)} paragraphs")
```

### Pattern 3: Forward-Looking Statement Extraction

```python
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# Initialize model
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-fls', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-fls')
nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer)

# Extract FLS
def extract_forward_looking_statements(sentences, specific_only=True):
    fls = []

    for sentence in sentences:
        result = nlp(sentence)
        label = result[0]['label']
        score = result[0]['score']

        if specific_only:
            if label == 'Specific FLS' and score >= 0.6:
                fls.append({
                    'text': sentence,
                    'type': label,
                    'confidence': score
                })
        else:
            if 'FLS' in label and score >= 0.6:
                fls.append({
                    'text': sentence,
                    'type': label,
                    'confidence': score
                })

    return fls

# Use it
sentences = ["sentence 1...", "sentence 2..."]
fls_statements = extract_forward_looking_statements(sentences)
print(f"Found {len(fls_statements)} forward-looking statements")
```

## Best Practices

### 1. Model Selection
- **Sentiment only**: Use `finbert-tone` (fastest, most accurate for sentiment)
- **ESG screening**: Use `finbert-esg` (4 categories, quick filtering)
- **Detailed ESG**: Use `finbert-esg-9-categories` (comprehensive ESG analysis)
- **FLS detection**: Use `finbert-fls` (specialized for forward-looking statements)
- **Chinese text**: Use `finbert-tone-chinese` (optimized for Chinese)

### 2. Text Preprocessing
- Keep financial context intact
- Maintain sentence boundaries
- Don't remove industry-specific terms
- Preserve numerical data and percentages

### 3. Confidence Thresholds
- **High confidence** (>0.8): Reliable for automated decisions
- **Medium confidence** (0.5-0.8): Good for filtering, may need review
- **Low confidence** (<0.5): Manual review recommended

### 4. Batch Processing
```python
# Process in batches for efficiency
results = nlp(sentences_list, batch_size=16)
```

### 5. Error Handling
```python
try:
    results = nlp(text)
except Exception as e:
    print(f"Error processing text: {e}")
    # Handle truncation for very long texts
    text = text[:512]  # BERT max length
    results = nlp(text)
```

## Performance Considerations

### Model Loading
- Models are ~440MB each
- First run downloads from HuggingFace
- Cache models locally for faster subsequent loads
- Consider loading once and reusing for multiple analyses

### Processing Speed
- **Single sentence**: ~50-100ms
- **Batch of 10**: ~200-300ms
- **Document (100 sentences)**: ~2-5 seconds
- Use GPU for faster processing (10-50x speedup)

### Memory Usage
- Each model: ~2GB RAM
- Multiple models: Load only what you need
- Clear models when switching:
```python
del finbert, tokenizer, nlp
import gc
gc.collect()
```

## Troubleshooting

### Model Download Issues
**Problem**: Can't download model from HuggingFace

**Solution**:
1. Check internet connection
2. Try: `export HF_ENDPOINT=https://hf-mirror.com` (China)
3. Download manually and load from local path

### Out of Memory
**Problem**: Model loading fails due to memory

**Solution**:
1. Close other applications
2. Process smaller batches
3. Use CPU instead of GPU initially
4. Load one model at a time

### Low Confidence Scores
**Problem**: All predictions have low confidence

**Solution**:
1. Check text is financial domain
2. Ensure proper sentence boundaries
3. Text may be too short or too long
4. Try preprocessing (remove special chars, etc.)

### Wrong Predictions
**Problem**: Model gives unexpected results

**Solution**:
1. Verify you're using correct model for task
2. Check label mappings (LABEL_0, LABEL_1, etc.)
3. Review input text quality
4. Consider ensemble of multiple models

## Integration Examples

### With Pandas DataFrames
```python
import pandas as pd

df = pd.read_csv('financial_texts.csv')
df['sentiment'] = df['text'].apply(lambda x: nlp(x)[0]['label'])
df['confidence'] = df['text'].apply(lambda x: nlp(x)[0]['score'])
```

### With Text Files
```python
def analyze_file(file_path):
    with open(file_path, 'r') as f:
        text = f.read()

    sentences = text.split('. ')
    results = nlp(sentences)

    return results
```

### With APIs
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json['text']
    result = nlp(text)
    return jsonify(result)
```

## Citation

If you use FinBERT models in research, please cite:

Huang, Allen H., Hui Wang, and Yi Yang. "FinBERT: A Large Language Model for Extracting Information from Financial Text." *Contemporary Accounting Research* (2022).

Yang, Yi, Mark Christopher Siy Uy, and Allen Huang. "Finbert: A pretrained language model for financial communications." *arXiv preprint arXiv:2006.08097* (2020).

## Resources

- **HuggingFace Models**: Search "yiyanghkust/finbert" on [huggingface.co](https://huggingface.co)
- **ESG Categories PDF**: [9-class descriptions](https://www.allenhuang.org/uploads/2/6/5/5/26555246/esg_9-class_descriptions.pdf)

## When to Use This Skill

Use this skill when you need to:
- Analyze sentiment in earnings calls, 10-Ks, analyst reports
- Extract ESG-related content from corporate reports
- Classify ESG topics for sustainability analysis
- Identify forward-looking statements in MD&A sections
- Process Chinese financial documents
- Build financial NLP pipelines
- Research financial text characteristics
- Automate financial document analysis

This skill is particularly useful for financial analysts, ESG researchers, compliance teams, and data scientists working with financial text data.
