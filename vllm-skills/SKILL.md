---
name: vllm-skills
description: Expert guidance for working with vLLM - a fast, easy-to-use library for Large Language Model (LLM) inference and serving. Use when deploying LLMs, setting up inference servers, optimizing model serving, implementing distributed inference, or troubleshooting vLLM performance issues.
license: Apache 2.0
---

# vLLM Skills - LLM Inference and Serving Expert

## Overview

vLLM is a high-performance library for LLM inference and serving, developed at UC Berkeley's Sky Computing Lab and now a PyTorch Foundation hosted project. This skill provides comprehensive guidance for working with vLLM's codebase, deploying models, and optimizing performance.

**Core Innovation:** PagedAttention - a revolutionary memory management technique for KV (key-value) cache that achieves efficient memory utilization by treating KV cache like virtual memory with pages.

**Key Capabilities:**
- State-of-the-art throughput for LLM serving
- OpenAI-compatible API endpoints
- Support for 198+ model architectures
- Multi-modal LLMs, embedding models, and mixture-of-expert architectures
- Distributed inference with tensor/pipeline/data/expert parallelism
- Advanced optimizations: continuous batching, prefix caching, speculative decoding

## Quick Start

### Prerequisites
- Python 3.9+
- CUDA 12.1+ (for NVIDIA GPUs)
- PyTorch 2.9.0+
- 16GB+ RAM recommended
- GPU with 8GB+ VRAM for small models

### Basic Installation
```bash
# Install from PyPI
pip install vllm

# Or install from source (development)
cd /Users/jason/Projects/vllm
pip install -e .
```

### Simplest Usage
```python
from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(model="facebook/opt-125m")

# Generate text
prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

## Core Workflows

### 1. Offline Batch Inference

**Use when:** Processing large batches of text offline without streaming.

```python
from vllm import LLM, SamplingParams

# Configure sampling
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# Initialize LLM
llm = LLM(
    model="meta-llama/Llama-3-8B",
    tensor_parallel_size=2,  # Use 2 GPUs
    dtype="float16"
)

# Batch generation
prompts = ["Your prompt here..."] * 1000
outputs = llm.generate(prompts, sampling_params)
```

**Key parameters:**
- `model`: HuggingFace model name or local path
- `tensor_parallel_size`: Number of GPUs for tensor parallelism
- `dtype`: Precision (float16, bfloat16, float32)
- `gpu_memory_utilization`: Fraction of GPU memory to use (default: 0.9)

### 2. Online Serving with OpenAI-Compatible API

**Use when:** Deploying a production API server for real-time inference.

**Start the server:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B \
    --tensor-parallel-size 2 \
    --dtype float16 \
    --port 8000
```

**Query the server:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"
)

completion = client.chat.completions.create(
    model="meta-llama/Llama-3-8B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"}
    ]
)

print(completion.choices[0].message.content)
```

**Streaming response:**
```python
stream = client.chat.completions.create(
    model="meta-llama/Llama-3-8B",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

### 3. Distributed Inference (Multi-GPU)

**Use when:** Model is too large for single GPU or need maximum throughput.

**Tensor Parallelism** (split model layers across GPUs):
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-70B",
    tensor_parallel_size=4,  # Use 4 GPUs
    pipeline_parallel_size=1
)
```

**Pipeline Parallelism** (split model depth across GPUs):
```python
llm = LLM(
    model="meta-llama/Llama-3-70B",
    tensor_parallel_size=2,
    pipeline_parallel_size=2  # 2x2 = 4 GPUs total
)
```

**Best practices:**
- Use tensor parallelism first (better load balancing)
- Use pipeline parallelism for very large models
- Combine both for massive models (70B+)

### 4. Multi-Modal Inference

**Use when:** Working with vision-language models.

```python
from vllm import LLM, SamplingParams
from PIL import Image

# Initialize multi-modal LLM
llm = LLM(model="llava-hf/llava-1.5-7b-hf")

# Prepare inputs
image = Image.open("path/to/image.jpg")
prompt = "Describe this image in detail."

# Generate
outputs = llm.generate(
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image}
    },
    sampling_params=SamplingParams(temperature=0.2)
)

print(outputs[0].outputs[0].text)
```

### 5. LoRA Adapters for Multi-Tenant Serving

**Use when:** Serving multiple fine-tuned versions of the same base model.

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Initialize with LoRA support
llm = LLM(
    model="meta-llama/Llama-3-8B",
    enable_lora=True,
    max_lora_rank=64
)

# Generate with different LoRA adapters
outputs = llm.generate(
    ["Tell me about AI"] * 2,
    sampling_params=SamplingParams(temperature=0.8),
    lora_request=[
        LoRARequest("adapter1", 1, "/path/to/lora1"),
        LoRARequest("adapter2", 2, "/path/to/lora2")
    ]
)
```

## Project Structure Reference

### Key Directories

**Core Engine:**
- `/vllm/engine/` - LLM engine implementations (sync/async)
- `/vllm/v1/` - V1 architecture (1.7x speedup, zero-overhead prefix caching)
- `/vllm/attention/` - Attention mechanisms (PagedAttention, FlashAttention)

**Model Support:**
- `/vllm/model_executor/models/` - 198+ model implementations
- `/vllm/model_executor/layers/` - Neural network layers
- `/vllm/config/` - Configuration classes

**API & Serving:**
- `/vllm/entrypoints/llm.py` - High-level Python API
- `/vllm/entrypoints/openai/` - OpenAI-compatible server

**Distributed:**
- `/vllm/distributed/` - Multi-GPU communication
- `/vllm/executor/` - Execution backends

**Optimizations:**
- `/csrc/` - CUDA/C++ GPU kernels
- `/vllm/model_executor/layers/quantization/` - GPTQ, AWQ, INT4, FP8

**Examples & Tests:**
- `/examples/` - Usage examples
- `/tests/` - Comprehensive test suites

## Configuration Guide

### VllmConfig - Master Configuration

```python
from vllm import VllmConfig, ModelConfig, ParallelConfig, CacheConfig

config = VllmConfig(
    model_config=ModelConfig(
        model="meta-llama/Llama-3-8B",
        dtype="float16",
        max_model_len=4096
    ),
    parallel_config=ParallelConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=1
    ),
    cache_config=CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9
    )
)
```

### Key Configuration Classes

**ModelConfig** - Model-specific settings:
- `model`: Model name/path
- `dtype`: Precision (float16, bfloat16, float32)
- `max_model_len`: Maximum sequence length
- `quantization`: Quantization method (gptq, awq, fp8)

**ParallelConfig** - Distributed inference:
- `tensor_parallel_size`: Tensor parallelism degree
- `pipeline_parallel_size`: Pipeline parallelism degree
- `data_parallel_size`: Data parallelism degree

**CacheConfig** - KV cache management:
- `block_size`: Size of KV cache blocks
- `gpu_memory_utilization`: GPU memory fraction to use
- `swap_space`: CPU memory for swapping

**SchedulerConfig** - Request scheduling:
- `max_num_batched_tokens`: Maximum tokens per batch
- `max_num_seqs`: Maximum sequences in batch
- `max_model_len`: Maximum sequence length

## Common Use Cases

### 1. High-Throughput Serving

**Scenario:** Maximize throughput for production API serving.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B \
    --tensor-parallel-size 2 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching
```

**Optimizations enabled:**
- Large batch sizes for continuous batching
- Prefix caching for repeated prompts
- High GPU memory utilization

### 2. Low-Latency Serving

**Scenario:** Minimize latency for interactive applications.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B \
    --max-num-seqs 32 \
    --max-num-batched-tokens 2048 \
    --enforce-eager
```

**Trade-offs:**
- Smaller batches for faster processing
- Eager mode instead of CUDA graphs (lower overhead)

### 3. Large Model Deployment (70B+)

**Scenario:** Deploy Llama-3-70B on 4xA100 GPUs.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70B \
    --tensor-parallel-size 4 \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90
```

**Memory optimization:**
- Tensor parallelism across 4 GPUs
- Float16 precision
- Conservative memory utilization

### 4. Quantized Model Serving

**Scenario:** Serve quantized model for reduced memory footprint.

```python
from vllm import LLM, SamplingParams

# AWQ quantized model (4-bit)
llm = LLM(
    model="TheBloke/Llama-2-13B-AWQ",
    quantization="awq",
    dtype="float16"
)

# Or GPTQ quantized
llm = LLM(
    model="TheBloke/Llama-2-13B-GPTQ",
    quantization="gptq",
    dtype="float16"
)
```

**Benefits:**
- 3-4x memory reduction
- ~2x throughput increase
- Minimal accuracy loss

## Troubleshooting

### Out of Memory Errors

**Symptom:** CUDA out of memory during initialization.

**Solutions:**
1. Reduce `gpu_memory_utilization`:
   ```python
   llm = LLM(model="...", gpu_memory_utilization=0.8)
   ```

2. Use quantization:
   ```python
   llm = LLM(model="...", quantization="awq")
   ```

3. Increase tensor parallelism:
   ```python
   llm = LLM(model="...", tensor_parallel_size=2)
   ```

4. Reduce max sequence length:
   ```python
   llm = LLM(model="...", max_model_len=2048)
   ```

### Slow Inference Speed

**Symptom:** Lower throughput than expected.

**Solutions:**
1. Enable prefix caching:
   ```python
   llm = LLM(model="...", enable_prefix_caching=True)
   ```

2. Increase batch size:
   ```bash
   --max-num-batched-tokens 8192 --max-num-seqs 256
   ```

3. Use CUDA graphs (default, but verify):
   ```python
   # Avoid --enforce-eager flag
   llm = LLM(model="...")  # CUDA graphs enabled by default
   ```

4. Upgrade to V1 architecture:
   ```python
   # vLLM V1 offers 1.7x speedup
   # Set environment variable
   export VLLM_USE_V1=1
   ```

### Model Not Supported

**Symptom:** Model architecture not recognized.

**Solutions:**
1. Check supported models:
   ```bash
   ls /Users/jason/Projects/vllm/vllm/model_executor/models/
   # 198+ model files listed
   ```

2. Use similar architecture:
   ```python
   # Many models share architectures
   # E.g., most 7B models use Llama architecture
   llm = LLM(model="your-model", trust_remote_code=True)
   ```

3. Add custom model (advanced):
   - Create new file in `/vllm/model_executor/models/`
   - Register in model registry
   - Follow existing model patterns

### Distributed Inference Issues

**Symptom:** Multi-GPU setup fails or hangs.

**Solutions:**
1. Verify GPU availability:
   ```bash
   nvidia-smi
   ```

2. Check tensor parallel size matches GPUs:
   ```python
   # If you have 4 GPUs:
   llm = LLM(model="...", tensor_parallel_size=4)
   ```

3. Use Ray for debugging:
   ```bash
   # Start Ray explicitly
   ray start --head
   python -m vllm.entrypoints.openai.api_server ...
   ```

## Advanced Usage

### Custom Sampling Parameters

```python
from vllm import SamplingParams

sampling_params = SamplingParams(
    temperature=0.8,           # Randomness (0.0 = deterministic)
    top_p=0.95,                # Nucleus sampling
    top_k=50,                  # Top-k sampling
    max_tokens=512,            # Maximum generation length
    presence_penalty=0.5,      # Penalize repeated tokens
    frequency_penalty=0.5,     # Penalize frequent tokens
    stop=["END", "\n\n"],     # Stop sequences
    n=3,                       # Generate 3 completions
    best_of=5,                 # Sample 5, return best 3
    use_beam_search=True       # Use beam search
)
```

### Structured Output (JSON Mode)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3-8B")

# Force JSON output
sampling_params = SamplingParams(
    temperature=0.8,
    guided_decoding={"type": "json"}
)

outputs = llm.generate(
    "Generate a JSON object with name and age fields",
    sampling_params
)
```

### Prefix Caching for Repeated Prompts

**Use when:** System prompt or context is reused across requests.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-8B",
    enable_prefix_caching=True
)

system_prompt = "You are a helpful assistant..." * 1000  # Long prompt

# First request: computes KV cache
outputs1 = llm.generate(
    system_prompt + "What is AI?",
    SamplingParams()
)

# Second request: reuses cached KV for system_prompt
outputs2 = llm.generate(
    system_prompt + "What is ML?",
    SamplingParams()
)
# Much faster due to prefix caching!
```

### Speculative Decoding

**Use when:** Trade computation for lower latency.

```python
llm = LLM(
    model="meta-llama/Llama-3-70B",
    speculative_model="meta-llama/Llama-3-8B",  # Draft model
    num_speculative_tokens=5
)

# Small model generates draft tokens
# Large model verifies in parallel
# Achieves 2-3x speedup for generation
```

## Performance Benchmarking

### Measure Throughput

```bash
cd /Users/jason/Projects/vllm/benchmarks

python benchmark_throughput.py \
    --model meta-llama/Llama-3-8B \
    --input-len 512 \
    --output-len 128 \
    --num-prompts 1000 \
    --tensor-parallel-size 2
```

### Measure Latency

```bash
python benchmark_latency.py \
    --model meta-llama/Llama-3-8B \
    --input-len 512 \
    --output-len 128
```

### Compare with Baseline

```bash
# vLLM vs HuggingFace Transformers
python benchmark_serving.py \
    --backend vllm \
    --model meta-llama/Llama-3-8B \
    --dataset sharegpt
```

## Best Practices

### 1. Memory Management
- Start with `gpu_memory_utilization=0.9`, reduce if OOM
- Use quantization for large models on limited hardware
- Monitor memory with `nvidia-smi` during serving

### 2. Parallelism Strategy
- **Single GPU:** No parallelism needed
- **2-4 GPUs:** Tensor parallelism only
- **8+ GPUs:** Combine tensor + pipeline parallelism
- **Multi-node:** Use data parallelism + tensor parallelism

### 3. Production Deployment
- Use OpenAI-compatible API for easy client integration
- Enable prefix caching for system prompts
- Monitor with Ray dashboard: `http://localhost:8265`
- Set up load balancing for multiple instances
- Use Docker for reproducible deployments

### 4. Model Selection
- 7B models: Single GPU (A100, H100)
- 13B models: Single high-memory GPU or 2 GPUs
- 30B models: 2-4 GPUs with tensor parallelism
- 70B models: 4-8 GPUs with tensor parallelism
- 175B+ models: 8+ GPUs with tensor + pipeline parallelism

### 5. Development Workflow
- Test with small models first (125M, 1B)
- Use `/examples/` directory for reference implementations
- Run tests: `pytest tests/` before deploying changes
- Check docs: `/docs/` for detailed guides
- Monitor issues: https://github.com/vllm-project/vllm/issues

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI
from vllm import LLM, SamplingParams

app = FastAPI()
llm = LLM(model="meta-llama/Llama-3-8B")

@app.post("/generate")
async def generate(prompt: str):
    outputs = llm.generate(
        prompt,
        SamplingParams(temperature=0.8, max_tokens=100)
    )
    return {"text": outputs[0].outputs[0].text}
```

### LangChain Integration

```python
from langchain.llms import VLLM

llm = VLLM(
    model="meta-llama/Llama-3-8B",
    trust_remote_code=True,
    max_new_tokens=512,
    temperature=0.8
)

response = llm("Tell me about quantum computing")
print(response)
```

### Gradio Interface

```python
import gradio as gr
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3-8B")

def generate(prompt, temperature, max_tokens):
    outputs = llm.generate(
        prompt,
        SamplingParams(temperature=temperature, max_tokens=max_tokens)
    )
    return outputs[0].outputs[0].text

interface = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Slider(0, 1, value=0.8, label="Temperature"),
        gr.Slider(1, 512, value=100, label="Max Tokens")
    ],
    outputs=gr.Textbox(label="Generated Text")
)

interface.launch()
```

## Key Files Reference

### Entry Points
- `/vllm/entrypoints/llm.py` - High-level Python API (`LLM` class)
- `/vllm/entrypoints/openai/api_server.py` - OpenAI-compatible server

### Core Engine
- `/vllm/engine/llm_engine.py` - Synchronous engine
- `/vllm/engine/async_llm_engine.py` - Asynchronous engine
- `/vllm/v1/engine/` - V1 architecture (latest, fastest)

### Configuration
- `/vllm/config/vllm_config.py` - Master configuration
- `/vllm/config/model_config.py` - Model settings
- `/vllm/config/parallel_config.py` - Distributed settings

### Models
- `/vllm/model_executor/models/` - 198+ model implementations
- `/vllm/model_executor/model_loader/` - Model loading logic

### Documentation
- `/docs/` - Full documentation
- `/README.md` - Project overview
- `/examples/` - Usage examples

## Resources

**Repository:** /Users/jason/Projects/vllm

**Official Resources:**
- GitHub: https://github.com/vllm-project/vllm
- Documentation: https://vllm.readthedocs.io/
- Paper: SOSP 2023 - "Efficient Memory Management for Large Language Model Serving with PagedAttention"

**Community:**
- Discord: vLLM community server
- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: Q&A and community support

**Related Projects:**
- LMSYS Chatbot Arena (uses vLLM)
- Ray (distributed execution framework)
- HuggingFace Transformers (model source)

---

## When to Use This Skill

Invoke this skill when:
- Setting up LLM inference or serving infrastructure
- Deploying models to production with OpenAI-compatible APIs
- Optimizing model serving performance and throughput
- Implementing distributed inference across multiple GPUs
- Working with multi-modal models or LoRA adapters
- Troubleshooting vLLM configuration or performance issues
- Integrating vLLM with other frameworks (FastAPI, LangChain, etc.)
- Benchmarking LLM serving performance
- Contributing to the vLLM codebase
- Understanding PagedAttention and KV cache optimization

This skill provides expert-level guidance for all aspects of vLLM deployment, configuration, and optimization, based on comprehensive analysis of the vLLM codebase structure and documentation.
