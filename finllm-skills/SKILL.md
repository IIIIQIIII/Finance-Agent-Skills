---
name: finllm
description: Financial NLP toolkit with 5 tasks (FPB sentiment, FiQA entity sentiment, Headline classification, NER, FLS detection) across 3 LLM providers (Qwen, Kimi, GLM). Supports dual prompt strategy (short/long), numerical confidence scoring, and pip-installable package structure.
---

# FinLLM: Financial NLP Toolkit

A production-ready Python package for financial NLP tasks using multiple Large Language Models, providing standardized prompt templates and easy-to-use interfaces for sentiment analysis, entity recognition, and forward-looking statement detection.

## What This Skill Does

FinLLM provides 5 financial NLP tasks with multi-model support:

1. **FPB (Financial PhraseBank)** - Sentiment analysis (positive/neutral/negative)
2. **FiQA** - Target entity sentiment analysis
3. **Headline** - News classification (good/bad/unknown news)
4. **NER** - Named Entity Recognition (Organization, Person, Location)
5. **FLS** - Forward-Looking Statement detection with numerical confidence (0.0-1.0)

**Key Features**:
- **3 LLM providers**: Qwen3 Max, Kimi K2 Thinking, GLM 4.6
- **Dual prompt strategy**: Short (efficiency) vs Long (performance)
- **Numerical confidence scoring**: 0.0-1.0 scale for FLS task
- **Pip-installable**: Standard Python package structure
- **100% test coverage**: All models × all tasks × all prompts

## Prerequisites

1. **Python 3.8+**

2. **Install the package**:
   ```bash
   # Option 1: Editable install (for development)
   git clone https://github.com/IIIIQIIII/finllm.git
   cd finllm
   pip install -e .

   # Option 2: From PyPI (when published)
   pip install finllm
   ```

3. **Configure API keys** in `.env` file:
   ```bash
   # Copy sample configuration
   cp .env.sample .env

   # Edit .env and add your API key
   # Required: DashScope API key for Aliyun models
   DASHSCOPE_API_KEY=your_api_key_here
   ```

## How to Use

### Basic Usage

**FPB Sentiment Analysis**:
```python
from finllm import QwenModel, FPBTask

model = QwenModel()
task = FPBTask(model, prompt_version="short")

result = task.classify("Company profits soared 50% this quarter")
print(result["sentiment"])  # Output: positive
```

**FiQA Target Entity Sentiment**:
```python
from finllm import KimiModel, FiQATask

model = KimiModel()
task = FiQATask(model, prompt_version="long")

text = "Apple's iPhone sales exceeded expectations"
result = task.classify(text)
print(f"{result['target']}: {result['sentiment']}")
# Output: Apple: positive
```

**Headline Classification**:
```python
from finllm import GLMModel, HeadlineTask

model = GLMModel()
task = HeadlineTask(model)

headline = "Stock market crashes amid economic uncertainty"
result = task.classify(headline)
print(result["classification"])  # Output: bad news
```

**Named Entity Recognition**:
```python
from finllm import QwenModel, NERTask

model = QwenModel()
task = NERTask(model, prompt_version="short")

text = "Tesla, Inc. CEO Elon Musk announced expansion in Austin, Texas."
result = task.extract_entities(text)

for entity in result['entities']:
    print(f"{entity['entity']} ({entity['type']})")
# Output:
# Tesla, Inc. (Organization)
# Elon Musk (Person)
# Austin, Texas (Location)
```

**Forward-Looking Statement Detection**:
```python
from finllm import QwenModel, FLSTask

model = QwenModel()
task = FLSTask(model, prompt_version="short")

text = "We expect production to increase 25% next year."
result = task.classify(text)
print(f"{result['classification']}: {result['confidence']:.2f}")
# Output: FLS: 0.95
```

### Batch Processing

All tasks support batch processing for efficiency:

```python
from finllm import QwenModel, FPBTask

model = QwenModel()
task = FPBTask(model, prompt_version="short")

texts = [
    "Revenue increased significantly",
    "Company facing bankruptcy",
    "Stock price unchanged"
]

results = task.classify_batch(texts)
for r in results:
    print(f"{r['sentiment']:8} | {r['text']}")
# Output:
# positive | Revenue increased significantly
# negative | Company facing bankruptcy
# neutral  | Stock price unchanged
```

### Comparing Models

Test the same task across different models:

```python
from finllm import QwenModel, KimiModel, GLMModel, FPBTask

text = "Stock prices jumped on positive earnings"

for ModelClass in [QwenModel, KimiModel, GLMModel]:
    model = ModelClass()
    task = FPBTask(model, prompt_version="short")
    result = task.classify(text)
    print(f"{model.model_id}: {result['sentiment']}")
```

### Prompt Version Comparison

Compare short vs long prompt performance:

```python
from finllm import QwenModel, NERTask

model = QwenModel()
text = "Apple Inc. announced new products in Cupertino."

for version in ["short", "long"]:
    task = NERTask(model, prompt_version=version)
    result = task.extract_entities(text)
    print(f"\n{version.upper()} prompt:")
    print(f"  Entities found: {len(result['entities'])}")
```

## Available Models

### Qwen3 Max (Fastest)
```python
from finllm import QwenModel
model = QwenModel()
# - Provider: Aliyun DashScope
# - Speed: 2.3s/sample (fastest)
# - Best for: Production, batch processing
```

### Kimi K2 Thinking (With Reasoning)
```python
from finllm import KimiModel
model = KimiModel()
# - Provider: Moonshot AI
# - Speed: 2.9s/sample
# - Best for: Complex analysis, explainability
```

### GLM 4.6 (Deep Analysis)
```python
from finllm import GLMModel
model = GLMModel()
# - Provider: Zhipu AI
# - Speed: 3.1s/sample
# - Best for: Thorough analysis, research
```

## Task Details

### 1. FPB - Sentiment Analysis

**Labels**: positive, neutral, negative

**Example**:
```python
from finllm import QwenModel, FPBTask

task = FPBTask(QwenModel())
result = task.classify("Earnings surpassed analyst expectations")
# {'sentiment': 'positive', 'text': '...', 'raw_output': '...'}
```

### 2. FiQA - Entity Sentiment

**Output**: Target entity + sentiment

**Example**:
```python
from finllm import QwenModel, FiQATask

task = FiQATask(QwenModel())
result = task.classify("Microsoft's cloud revenue grew 30%")
# {'target': 'Microsoft', 'sentiment': 'positive', ...}
```

### 3. Headline - News Classification

**Labels**: good news, bad news, unknown

**Example**:
```python
from finllm import QwenModel, HeadlineTask

task = HeadlineTask(QwenModel())
result = task.classify("Company announces massive layoffs")
# {'classification': 'bad news', ...}
```

### 4. NER - Named Entity Recognition

**Entity Types**: Organization, Person, Location

**Example**:
```python
from finllm import QwenModel, NERTask

task = NERTask(QwenModel())
result = task.extract_entities("JPMorgan Chase CEO Jamie Dimon spoke in New York")
# {'entities': [
#   {'entity': 'JPMorgan Chase', 'type': 'Organization'},
#   {'entity': 'Jamie Dimon', 'type': 'Person'},
#   {'entity': 'New York', 'type': 'Location'}
# ]}
```

### 5. FLS - Forward-Looking Statement

**Labels**: FLS, NOT_FLS
**Confidence**: 0.0-1.0 numerical score

**Example**:
```python
from finllm import QwenModel, FLSTask

task = FLSTask(QwenModel())

# Future-oriented statement
result1 = task.classify("We plan to expand operations next quarter")
# {'classification': 'FLS', 'confidence': 0.92, ...}

# Historical statement
result2 = task.classify("Revenue increased 20% last year")
# {'classification': 'NOT_FLS', 'confidence': 0.88, ...}
```

**Confidence Scale**:
- **0.90-1.00**: Very clear signal
- **0.75-0.89**: Clear signal
- **0.60-0.74**: Moderate confidence
- **Below 0.60**: Low confidence

## Configuration

### Customizing Prompts

Prompts are stored in `prompts/{task}/{version}.txt`:

```
prompts/
├── fpb/
│   ├── short.txt
│   └── long.txt
├── fiqa/
│   ├── short.txt
│   └── long.txt
├── headline/
│   ├── short.txt
│   └── long.txt
├── ner/
│   ├── short.txt
│   └── long.txt
└── fls/
    ├── short.txt
    └── long.txt
```

You can modify these files to customize prompts for your use case.

### Model Configuration

Models are configured in `config/models.json`:

```json
{
  "qwen": {
    "provider": "dashscope",
    "model_id": "qwen-max",
    "api_key_env": "DASHSCOPE_API_KEY"
  },
  "kimi": {
    "provider": "moonshot",
    "model_id": "moonshot-v1-128k",
    "api_key_env": "MOONSHOT_API_KEY"
  },
  "glm": {
    "provider": "zhipuai",
    "model_id": "glm-4",
    "api_key_env": "ZHIPUAI_API_KEY"
  }
}
```

## Package Structure

```
finllm/
├── finllm/              # Main package
│   ├── __init__.py     # Top-level API
│   ├── models/         # Model integrations
│   │   ├── qwen.py
│   │   ├── kimi.py
│   │   └── glm.py
│   ├── tasks/          # Task implementations
│   │   ├── fpb.py
│   │   ├── fiqa.py
│   │   ├── headline.py
│   │   ├── ner.py
│   │   └── fls.py
│   └── utils.py
├── prompts/            # Prompt templates
├── config/             # Configuration files
├── examples/           # Example scripts
├── tests/              # Test data
├── pyproject.toml      # Package metadata
└── README.md           # Documentation
```

## Output Format

All tasks return dictionaries with consistent structure:

**FPB, FiQA, Headline**:
```python
{
  'text': 'input text',
  'sentiment': 'positive',      # or 'classification'/'target'
  'raw_output': 'LLM raw output'
}
```

**NER**:
```python
{
  'text': 'input text',
  'entities': [
    {'entity': 'Tesla', 'type': 'Organization'},
    {'entity': 'Elon Musk', 'type': 'Person'}
  ],
  'raw_output': 'LLM raw output'
}
```

**FLS**:
```python
{
  'text': 'input text',
  'classification': 'FLS',       # or 'NOT_FLS'
  'confidence': 0.95,            # 0.0-1.0
  'raw_output': 'LLM raw output'
}
```

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'finllm'`

**Solution**:
```bash
# Ensure package is installed
pip list | grep finllm

# If not installed, install in editable mode
pip install -e .
```

### API Key Errors

**Problem**: `AuthenticationError` or `No API key found`

**Solution**:
1. Check `.env` file exists in project root
2. Verify API key is correct
3. Ensure key is not expired

```bash
# Check .env file
cat .env

# Should show:
# DASHSCOPE_API_KEY=sk-...
```

### Model Not Responding

**Problem**: Timeout or no response

**Solution**:
1. Check internet connection
2. Verify API key has access to the model
3. Try a different model:
   ```python
   # If Qwen fails, try Kimi
   from finllm import KimiModel
   model = KimiModel()
   ```

### Parsing Errors

**Problem**: Task returns `None` or empty results

**Solution**:
1. Check raw output: `result['raw_output']`
2. Try the long prompt version:
   ```python
   task = FPBTask(model, prompt_version="long")
   ```
3. Verify input text is appropriate for the task

## Best Practices

1. **Use short prompts for batch processing**: More efficient, lower cost
2. **Use long prompts for critical analysis**: Better accuracy, more context
3. **Handle errors gracefully**:
   ```python
   try:
       result = task.classify(text)
   except Exception as e:
       print(f"Error: {e}")
   ```
4. **Validate output**:
   ```python
   if result and result.get('sentiment'):
       # Process result
   else:
       # Handle failure
   ```
5. **Monitor API costs**: Different models have different pricing
6. **Cache results**: Avoid re-processing same text

## Examples

### Example 1: Quick Sentiment Check
```python
from finllm import QwenModel, FPBTask

model = QwenModel()
task = FPBTask(model)
result = task.classify("Stocks are up 10% today")
print(result['sentiment'])  # positive
```

### Example 2: Batch Entity Recognition
```python
from finllm import QwenModel, NERTask

model = QwenModel()
task = NERTask(model)

texts = [
    "Apple CEO Tim Cook spoke at the event",
    "Microsoft acquired LinkedIn",
    "Amazon opened a new warehouse in Seattle"
]

results = task.extract_entities_batch(texts)
for r in results:
    print(f"\n{r['text']}")
    for e in r['entities']:
        print(f"  - {e['entity']} ({e['type']})")
```

### Example 3: FLS Detection Pipeline
```python
from finllm import QwenModel, FLSTask

model = QwenModel()
task = FLSTask(model, prompt_version="long")

# Process multiple statements
statements = [
    "We expect revenue growth of 15% in Q4",
    "Last quarter revenue was $100M",
    "The company plans to expand into new markets",
    "Historical performance has been strong"
]

fls_statements = []
for stmt in statements:
    result = task.classify(stmt)
    if result['classification'] == 'FLS' and result['confidence'] > 0.8:
        fls_statements.append({
            'text': stmt,
            'confidence': result['confidence']
        })

print(f"Found {len(fls_statements)} forward-looking statements")
for fls in fls_statements:
    print(f"  [{fls['confidence']:.2f}] {fls['text']}")
```

## When to Use This Skill

Use FinLLM when you need to:
- **Analyze financial text sentiment** at scale
- **Extract entities** from financial documents
- **Classify news headlines** (good/bad/unknown)
- **Detect forward-looking statements** in earnings calls or reports
- **Compare multiple LLM providers** for financial NLP tasks
- **Build financial NLP pipelines** with standardized interfaces

This skill is particularly useful for:
- Financial analysts and researchers
- Quantitative traders and portfolio managers
- Fintech developers building NLP features
- Academic researchers studying financial sentiment
- Data scientists analyzing financial documents

## Related Resources

- **GitHub Repository**: https://github.com/IIIIQIIII/finllm
- **PyPI Package**: (to be published)
- **Dataset Sources**: AdaptLLM on HuggingFace
- **Model Providers**:
  - Qwen: Aliyun DashScope
  - Kimi: Moonshot AI
  - GLM: Zhipu AI

## Version Information

**Current Version**: 1.2.0

**Recent Updates**:
- Added FLS task with numerical confidence scoring
- Restructured as pip-installable package
- Updated all documentation
- 100% test coverage across all models

## Support

For issues, questions, or contributions:
- Open an issue on GitHub: https://github.com/IIIIQIIII/finllm/issues
- Check documentation: README.md, QUICKSTART.md, INSTALLATION.md
- Review examples in `examples/` directory
