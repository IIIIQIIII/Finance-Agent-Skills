---
name: sentence-segmentation
description: A Python toolkit for sentence segmentation with unified API supporting NLTK, spaCy, PySBD, and Stanza frameworks. Easily split text into sentences using multiple NLP libraries through a consistent interface.
---

# Sentence Segmentation Toolkit

A comprehensive Python toolkit for sentence boundary detection (sentence segmentation) using multiple popular NLP frameworks through a unified API.

## What This Skill Does

This skill helps you segment text into sentences using four different NLP frameworks:

1. **NLTK** - Classic, fast, and reliable sentence tokenization
2. **spaCy** - Industrial-strength NLP with excellent accuracy
3. **PySBD** - Rule-based segmentation with no model downloads required
4. **Stanza** - Stanford NLP's neural network approach for highest accuracy

The toolkit provides a **unified API** that lets you switch between frameworks easily, making it simple to compare results or choose the best framework for your use case.

## Prerequisites

Before using this skill, ensure:

1. **Python 3.8 or higher** is installed
2. **uv** (recommended) or pip for package management

## How to Use

### Installation

#### Method 1: Clone from GitHub

```bash
# Clone the repository
git clone https://github.com/IIIIQIIII/sentence-segmentation.git
cd sentence-segmentation

# Create virtual environment
uv venv
# Or: python -m venv .venv

# Activate environment
source activate.sh
# Or: source .venv/bin/activate

# Install in editable mode
uv pip install -e .
# Or: pip install -e .
```

#### Method 2: Quick Setup

```bash
cd sentence-segmentation
uv venv
source activate.sh
uv pip install -r requirements.txt
```

### Download NLP Models

After installation, download the required models:

```bash
# NLTK punkt tokenizer (small, fast)
python -c "import nltk; nltk.download('punkt')"

# spaCy English model (medium size, high accuracy)
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# Stanza English model (large, highest accuracy)
python -c "import stanza; stanza.download('en')"
```

**Note**: PySBD works immediately without any model downloads!

### Basic Usage

```python
from sentence_segmentation import SentenceSegmenter

# Choose your framework: 'pysbd', 'nltk', 'spacy', or 'stanza'
segmenter = SentenceSegmenter(framework='pysbd')

# Segment text into sentences
text = "Hello world. This is a test. How are you?"
sentences = segmenter.segment(text)

print(sentences)
# Output: ['Hello world. ', 'This is a test. ', 'How are you?']
```

### Comparing Frameworks

```python
from sentence_segmentation import SentenceSegmenter

text = "Dr. Smith went to Washington D.C. He met with Prof. Johnson."

for framework in ['pysbd', 'nltk', 'spacy', 'stanza']:
    segmenter = SentenceSegmenter(framework=framework)
    sentences = segmenter.segment(text)
    print(f"\n{framework.upper()}: {len(sentences)} sentences")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. {sent.strip()}")
```

### Performance Benchmarking

```python
from sentence_segmentation import SentenceSegmenter

segmenter = SentenceSegmenter(framework='pysbd')
text = "Your text here. Multiple sentences."

# Benchmark the segmentation speed
results = segmenter.benchmark(text, iterations=1000)

print(f"Framework: {results['framework']}")
print(f"Average time: {results['average_time']*1000:.2f} ms")
print(f"Throughput: {results['sentences_per_second']:.0f} segments/sec")
```

## Output

### Example Output

```python
from sentence_segmentation import SentenceSegmenter

text = """Dr. Smith works at the U.S. Department of Health.
He specializes in A.I. research. His email is john@example.com."""

segmenter = SentenceSegmenter(framework='pysbd')
sentences = segmenter.segment(text)

for i, sentence in enumerate(sentences, 1):
    print(f"{i}. {sentence.strip()}")
```

**Output:**
```
1. Dr. Smith works at the U.S. Department of Health.
2. He specializes in A.I. research.
3. His email is john@example.com.
```

### Performance Comparison

| Framework | Speed (segments/sec) | Accuracy | Model Download |
|-----------|---------------------|----------|----------------|
| NLTK      | ~14,000             | Good     | Small (~2MB)   |
| PySBD     | ~1,500              | Good     | None required  |
| spaCy     | ~90                 | Excellent| Medium (~12MB) |
| Stanza    | ~85                 | Excellent| Large (~500MB) |

## Understanding Results

### Framework Characteristics

**NLTK**:
- ✓ Fastest processing speed
- ✓ Lightweight and reliable
- ✗ May split incorrectly on some abbreviations
- Best for: High-throughput applications

**PySBD**:
- ✓ No model download required
- ✓ Excellent handling of URLs, emails, abbreviations
- ✓ Good balance of speed and accuracy
- Best for: Quick setup, general use

**spaCy**:
- ✓ Industrial-grade accuracy
- ✓ Good with corporate names and abbreviations
- ✓ Fast after model loading
- Best for: Production environments

**Stanza**:
- ✓ Highest accuracy (neural network-based)
- ✓ Excellent with complex text
- ✗ Slower processing speed
- Best for: Research, when accuracy is critical

## Examples

### Example 1: Quick Start with PySBD

```bash
cd sentence-segmentation
source activate.sh
python examples/pysbd_example.py
```

### Example 2: Processing a Document

```python
from sentence_segmentation import SentenceSegmenter

# Read document
with open('document.txt', 'r') as f:
    text = f.read()

# Segment using spaCy
segmenter = SentenceSegmenter(framework='spacy')
sentences = segmenter.segment(text)

# Save results
with open('sentences.txt', 'w') as f:
    for i, sentence in enumerate(sentences, 1):
        f.write(f"{i}. {sentence.strip()}\n")

print(f"Extracted {len(sentences)} sentences")
```

### Example 3: Batch Processing

```python
from sentence_segmentation import SentenceSegmenter
from pathlib import Path

segmenter = SentenceSegmenter(framework='pysbd')

# Process multiple files
for txt_file in Path('documents/').glob('*.txt'):
    with open(txt_file) as f:
        text = f.read()

    sentences = segmenter.segment(text)

    # Save with same name
    output = txt_file.stem + '_sentences.txt'
    with open(f'output/{output}', 'w') as f:
        f.write('\n'.join(sentences))

    print(f"Processed {txt_file.name}: {len(sentences)} sentences")
```

### Example 4: Multilingual Support

```python
from sentence_segmentation import SentenceSegmenter

# Initialize with language parameter
segmenter = SentenceSegmenter(framework='stanza', language='zh')  # Chinese
# Or: language='es' for Spanish, 'fr' for French, etc.

text = "你的中文文本。这是第二句话。"
sentences = segmenter.segment(text)
```

## Troubleshooting

### "Model not found" Error

**Problem**: NLP model hasn't been downloaded

**Solution**: Download the required model:
```bash
# For NLTK
python -c "import nltk; nltk.download('punkt')"

# For spaCy
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# For Stanza
python -c "import stanza; stanza.download('en')"
```

### Import Error

**Problem**: Package not installed or environment not activated

**Solution**:
1. Check environment: `which python`
2. Activate environment: `source activate.sh`
3. Reinstall: `uv pip install -e .`

### Poor Segmentation Quality

**Problem**: Framework not handling your text well

**Solution**: Try a different framework:
- For abbreviations-heavy text: Use PySBD or spaCy
- For general text: Use NLTK for speed, spaCy for accuracy
- For academic/complex text: Use Stanza

### Slow Performance

**Problem**: Processing is too slow

**Solution**:
- Use NLTK for fastest speed (~14k segments/sec)
- Use PySBD for good balance (~1.5k segments/sec)
- Avoid Stanza for large volumes (unless accuracy is critical)

## Advanced Usage

### Custom Language Models

```python
from sentence_segmentation import SentenceSegmenter

# Use custom spaCy model
segmenter = SentenceSegmenter(framework='spacy', language='en')

# For other languages
segmenter_es = SentenceSegmenter(framework='stanza', language='es')
segmenter_zh = SentenceSegmenter(framework='stanza', language='zh')
```

### Integration with Other Tools

```python
import pandas as pd
from sentence_segmentation import SentenceSegmenter

# Process DataFrame
df = pd.read_csv('articles.csv')
segmenter = SentenceSegmenter(framework='pysbd')

df['sentences'] = df['text'].apply(lambda x: segmenter.segment(x))
df['sentence_count'] = df['sentences'].apply(len)

df.to_csv('articles_segmented.csv', index=False)
```

### Performance Optimization

```python
from sentence_segmentation import SentenceSegmenter

# Initialize once, reuse many times
segmenter = SentenceSegmenter(framework='spacy')

# Process many texts
texts = ["Text 1...", "Text 2...", "Text 3..."]
all_sentences = [segmenter.segment(text) for text in texts]

# This is faster than recreating the segmenter each time
```

## Best Practices

1. **Choose the right framework**:
   - PySBD: Start here (no downloads, works immediately)
   - NLTK: Need high speed
   - spaCy: Production environments
   - Stanza: Research/accuracy-critical work

2. **Initialize once**: Create the segmenter once and reuse it

3. **Benchmark your use case**: Test different frameworks on your actual data

4. **Handle errors gracefully**: Some frameworks may fail on certain inputs

5. **Consider memory**: Stanza and spaCy load models into memory (~100MB+)

6. **Cache models**: Models are loaded once per segmenter instance

7. **Test edge cases**: Abbreviations, URLs, emails, quotes

## When to Use This Skill

Use this skill when you need to:
- **Preprocess text** for NLP pipelines
- **Split documents** into sentence-level data
- **Analyze text** at sentence granularity
- **Compare segmentation** approaches
- **Build text processing** applications
- **Extract sentences** from documents, articles, or reports
- **Prepare training data** for machine learning models
- **Analyze sentence structure** and length

This skill is particularly useful for NLP engineers, data scientists, researchers, and developers working with text processing tasks.

## Related Files

- **Main code**: `src/sentence_segmentation/segmenter.py`
- **Examples**: `examples/*.py`
- **Tests**: `tests/test_segmenter.py`
- **Documentation**: `README.md`, `QUICKSTART.md`, `INSTALL.md`
- **Project repository**: https://github.com/IIIIQIIII/sentence-segmentation

## Verification

To verify the installation works:

```bash
# Run verification script
python verify_setup.py

# Or run demo
python demo_installed_package.py

# Or test manually
python -c "from sentence_segmentation import SentenceSegmenter; print('✓ Installation successful!')"
```

## Framework Selection Guide

**Choose NLTK when**:
- You need maximum speed
- Text is relatively simple
- You're processing large volumes

**Choose PySBD when**:
- You want quick setup (no downloads)
- Text has URLs, emails, abbreviations
- You need good accuracy with minimal setup

**Choose spaCy when**:
- You're building production systems
- You need consistent, high accuracy
- You have moderate performance requirements

**Choose Stanza when**:
- Accuracy is the top priority
- You're doing research or analysis
- Speed is not critical
- You need multilingual support

## Performance Tips

1. **Batch processing**: Process multiple texts without reinitializing
2. **Use NLTK for preprocessing**: Then use spaCy/Stanza for final analysis
3. **Profile your pipeline**: Use the `benchmark()` method
4. **Consider text size**: Larger texts benefit more from faster frameworks
5. **Cache segmenter instances**: Don't create new ones for each text
