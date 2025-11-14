---
name: llama-factory-skills
description: Guide for fine-tuning and training large language models using LLaMA-Factory. Use when users want to train, fine-tune, evaluate, or deploy LLMs locally, including tasks like supervised fine-tuning (SFT), LoRA adaptation, RLHF, preference optimization (DPO/KTO), or model deployment.
---

# LLaMA-Factory: LLM Training and Fine-Tuning Platform

## Overview

LLaMA-Factory is a simple, easy-to-use, and efficient platform for training and fine-tuning large language models. It enables fine-tuning of 100+ pre-trained models locally without writing code, supporting various training methods from basic supervised fine-tuning to advanced RLHF techniques.

**Key Capabilities:**
- **Models**: LLaMA, Mistral, Qwen, Gemma, ChatGLM, Phi, Yi, Baichuan, and 100+ more
- **Training Methods**: Pre-training, SFT, LoRA, RLHF (PPO), DPO, KTO, ORPO
- **Fine-tuning Techniques**: Full parameter, Freeze, LoRA variants (LoRA+, rsLoRA, DoRA, PiSSA), GaLore, BAdam
- **Quantization**: 2/3/4/5/6/8-bit QLoRA, GPTQ, AWQ, AQLM, bitsandbytes, HQQ, EETQ
- **Acceleration**: FlashAttention-2, Unsloth, Liger Kernel
- **Deployment**: OpenAI-compatible API, vLLM high-throughput serving
- **Interfaces**: WebUI (zero-code), CLI, Python API

## Core Concepts

### Training Stages

| Stage | Purpose | Input Data |
|-------|---------|-----------|
| **pt** (pre-training) | Learn language representations from scratch | Unlabeled text corpus |
| **sft** (supervised fine-tuning) | Adapt model to specific tasks | Instruction-response pairs |
| **rm** (reward modeling) | Train reward model for RLHF | Ranked responses |
| **ppo** | Optimize with reinforcement learning | Prompts + reward model |
| **dpo** | Direct preference optimization (no reward model) | Preferred vs rejected pairs |
| **kto** | Binary feedback optimization | Liked/disliked responses |

### Fine-tuning Types

| Type | Memory | Speed | Quality | Use Case |
|------|--------|-------|---------|----------|
| **full** | Highest | Slowest | Best | Small models, high GPU memory |
| **freeze** | Medium | Medium | Good | Freeze base layers, train head |
| **lora** | Lowest | Fastest | Very Good | Most common, 75% memory reduction |

### Data Formats

**Alpaca Format** (recommended for most tasks):
```json
[
  {
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "The capital of France is Paris."
  }
]
```

**ShareGPT Format** (for multi-turn conversations):
```json
[
  {
    "conversations": [
      {"from": "human", "value": "Hello!"},
      {"from": "gpt", "value": "Hi! How can I help you?"},
      {"from": "human", "value": "Tell me a joke."},
      {"from": "gpt", "value": "Why did the..."}
    ]
  }
]
```

## Getting Started Workflow

### 1. Installation

```bash
# Install CUDA dependencies (for GPU)
# Check: nvidia-smi

# Install LLaMA-Factory
pip install llamafactory

# Verify installation
llamafactory-cli version
llamafactory-cli env
```

### 2. Data Preparation

**Step 1: Prepare your data file**
- Save as JSON/JSONL in Alpaca or ShareGPT format
- Place in `data/` directory

**Step 2: Register in dataset_info.json**
```json
{
  "my_dataset": {
    "file_name": "my_data.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

**Step 3: Validate format**
- Ensure JSON is properly formatted
- Check column names match your data structure
- Verify no missing required fields

### 3. WebUI Training (Zero-Code Approach)

```bash
# Launch WebUI
llamafactory-cli webui

# Navigate to http://localhost:7860
# Use GUI to:
# 1. Select model and dataset
# 2. Configure training parameters
# 3. Start training
# 4. Monitor real-time loss curves
# 5. Chat with trained model
# 6. Export final model
```

**WebUI Sections:**
- **Training**: Configure and launch training jobs
- **Evaluation**: Test model performance
- **Chat**: Interactive testing
- **Export**: Merge LoRA and export models

## Training Workflows

### Workflow 1: Basic Supervised Fine-Tuning (SFT) with LoRA

**When to use**: Adapting a pre-trained model to specific instructions or tasks with limited GPU memory.

**Configuration (sft_lora.yaml)**:
```yaml
### Model
model_name_or_path: meta-llama/Llama-3-8B-Instruct
stage: sft
finetuning_type: lora

### Dataset
dataset: my_dataset
template: llama3
cutoff_len: 2048
max_samples: 10000

### Output
output_dir: saves/llama3-8b-sft-lora
overwrite_output_dir: true

### Training hyperparameters
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1

### Optimization
bf16: true
fp16: false
gradient_checkpointing: true
flash_attn: fa2

### LoRA
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target: all

### Evaluation
val_size: 0.1
per_device_eval_batch_size: 2
eval_strategy: steps
eval_steps: 500

### Logging
logging_steps: 10
save_steps: 500
save_total_limit: 3
```

**Execute training**:
```bash
llamafactory-cli train sft_lora.yaml
```

**Monitor progress**:
- Watch console output for loss curves
- Or enable TensorBoard: `tensorboard --logdir saves/llama3-8b-sft-lora`
- Or use SwanLab/Wandb for cloud monitoring

### Workflow 2: Direct Preference Optimization (DPO)

**When to use**: Aligning model outputs with human preferences using preference pairs (without needing a separate reward model).

**Prerequisites**:
- A model fine-tuned with SFT (used as reference model)
- Preference dataset with chosen/rejected pairs

**Data format**:
```json
[
  {
    "instruction": "Write a poem about AI",
    "input": "",
    "output": ["Good response...", "Bad response..."]
  }
]
```

**Configuration (dpo.yaml)**:
```yaml
### Model
model_name_or_path: saves/llama3-8b-sft-lora  # SFT model
stage: dpo
finetuning_type: lora
ref_model: meta-llama/Llama-3-8B-Instruct  # Reference model

### Dataset
dataset: preference_data
template: llama3

### DPO parameters
pref_beta: 0.1  # KL penalty coefficient
pref_loss: sigmoid  # or "hinge", "ipo", "kto_pair"

### Training
learning_rate: 5.0e-6
num_train_epochs: 1.0
# ... other training params similar to SFT
```

### Workflow 3: Merging LoRA Adapters

**When to use**: Converting LoRA adapters back into a full model for deployment or distribution.

**Configuration (merge_lora.yaml)**:
```yaml
model_name_or_path: meta-llama/Llama-3-8B-Instruct
adapter_name_or_path: saves/llama3-8b-sft-lora
template: llama3
finetuning_type: lora
export_dir: models/llama3-8b-merged
export_size: 2
export_device: auto
export_legacy_format: false
```

**Execute merge**:
```bash
llamafactory-cli export merge_lora.yaml
```

**Result**: Full merged model saved to `models/llama3-8b-merged/`

### Workflow 4: Model Quantization (Post-Training)

**When to use**: Reducing model size for deployment with minimal quality loss.

**Configuration (quantization.yaml)**:
```yaml
model_name_or_path: models/llama3-8b-merged
template: llama3
export_dir: models/llama3-8b-gptq-4bit
export_size: 2
export_device: auto
export_quantization_bit: 4  # 2, 3, 4, 5, 6, or 8
export_quantization_dataset: c4  # Calibration dataset
```

**Execute quantization**:
```bash
llamafactory-cli export quantization.yaml
```

## Inference and Deployment

### Interactive Chat (Command Line)

```bash
llamafactory-cli chat chat_config.yaml
```

**Configuration (chat_config.yaml)**:
```yaml
model_name_or_path: models/llama3-8b-merged
template: llama3
finetuning_type: lora  # If using LoRA adapter
adapter_name_or_path: saves/llama3-8b-sft-lora  # Optional
```

### Web Chat Interface

```bash
llamafactory-cli webchat chat_config.yaml
```

Access at `http://localhost:8000`

### API Deployment (OpenAI-Compatible)

```bash
# Set environment variables
export API_PORT=8000
export API_KEY=your_secret_key  # Optional

# Launch API server
llamafactory-cli api api_config.yaml
```

**Configuration (api_config.yaml)**:
```yaml
model_name_or_path: models/llama3-8b-merged
template: llama3
infer_backend: huggingface  # or "vllm" for high throughput
```

**Test API**:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_secret_key" \
  -d '{
    "model": "llama3-8b-merged",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### vLLM Deployment (High Throughput)

**When to use**: Production environments requiring high concurrent request handling.

```yaml
model_name_or_path: models/llama3-8b-merged
template: llama3
infer_backend: vllm
vllm_enforce_eager: true
vllm_max_model_len: 4096
```

**Benefits**:
- 10-20x higher throughput than standard inference
- Efficient GPU memory management
- PagedAttention optimization

## Evaluation

### Benchmark Evaluation (MMLU, CEVAL, CMMLU)

```yaml
model_name_or_path: models/llama3-8b-merged
stage: sft
finetuning_type: lora
adapter_name_or_path: saves/llama3-8b-sft-lora
template: llama3

task: mmlu  # or ceval, cmmlu
task_dir: evaluation
lang: en  # or zh
n_shot: 5
batch_size: 4
```

```bash
llamafactory-cli eval eval_config.yaml
```

### NLG Metrics (BLEU, ROUGE)

```yaml
model_name_or_path: models/llama3-8b-merged
stage: sft
finetuning_type: lora
adapter_name_or_path: saves/llama3-8b-sft-lora
template: llama3
dataset: eval_dataset
predict_with_generate: true
```

```bash
llamafactory-cli eval eval_config.yaml
```

## Advanced Configurations

### Distributed Training

**Single Node, Multiple GPUs**:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train config.yaml
```

**DeepSpeed ZeRO-3** (for large models):
```yaml
deepspeed: examples/deepspeed/ds_z3_config.json
```

**FSDP** (Fully Sharded Data Parallel):
```yaml
fsdp: full_shard auto_wrap
fsdp_config:
  fsdp_backward_prefetch: backward_pre
  fsdp_forward_prefetch: false
  fsdp_use_orig_params: true
```

### Memory Optimization Techniques

| Technique | Memory Saving | Configuration |
|-----------|--------------|---------------|
| Gradient checkpointing | 30-50% | `gradient_checkpointing: true` |
| FlashAttention-2 | 20-30% | `flash_attn: fa2` |
| LoRA fine-tuning | 75% | `finetuning_type: lora` |
| 4-bit quantization | 75% | `quantization_bit: 4` |
| DeepSpeed ZeRO-3 | 60-80% | `deepspeed: ds_z3_config.json` |
| Mixed precision (bf16) | 50% | `bf16: true` |

**Combine techniques for extreme memory reduction**:
```yaml
finetuning_type: lora
quantization_bit: 4
gradient_checkpointing: true
flash_attn: fa2
bf16: true
```

### Multi-Modal Fine-Tuning

For vision-language models (e.g., LLaVA):

```yaml
model_name_or_path: llava-hf/llava-1.5-7b-hf
stage: sft
finetuning_type: lora
template: llava  # Multi-modal template
dataset: mllm_dataset  # Must include image paths

# Ensure image token count matches
# <image> placeholder should match actual image tokens
```

## Best Practices

### 1. Model Selection
- Start with instruction-tuned variants (e.g., `Llama-3-8B-Instruct` not base `Llama-3-8B`)
- Match template to model family (`llama3`, `qwen`, `chatglm`, etc.)
- Consider model size vs available VRAM

### 2. Data Quality
- Clean and diverse training data (aim for 1000+ high-quality examples)
- Validate JSON format before training
- Use validation split (10-20%) to monitor overfitting
- Balance dataset across different task types

### 3. Hyperparameter Tuning
- **Learning rate**: Start with 1e-4 for LoRA, 1e-5 for full fine-tuning
- **Batch size**: Maximize within GPU memory (use gradient accumulation)
- **LoRA rank**: 8-16 for most tasks, 32-64 for complex domains
- **Epochs**: 1-3 for large datasets, 5-10 for small datasets
- **Warmup**: 5-10% of total steps

### 4. Training Monitoring
- Watch for loss plateaus (may indicate convergence or overfitting)
- Validate regularly on held-out set
- Use early stopping if validation loss increases
- Test chat quality during training (use WebUI or CLI chat)

### 5. Memory Management
- Enable `gradient_checkpointing` if OOM errors occur
- Reduce `cutoff_len` (max sequence length) to save memory
- Use smaller batch size with more `gradient_accumulation_steps`
- Try 4-bit quantization (`quantization_bit: 4`) for very large models

### 6. Deployment Optimization
- Merge LoRA adapters before deployment for faster inference
- Use vLLM backend for production API serving
- Quantize merged models for edge deployment
- Set appropriate `max_model_len` to match typical request sizes

### 7. Template Matching
Always match the template to the model:
- Llama-3: `template: llama3`
- Qwen2: `template: qwen`
- ChatGLM3: `template: chatglm3`
- Mistral: `template: mistral`
- Check model card for template format

### 8. Common Pitfalls
- **Wrong template**: Causes poor performance or formatting errors
- **Excessive learning rate**: Model diverges or generates gibberish
- **Too many epochs**: Overfitting, model memorizes training data
- **Mismatched data format**: Training fails or produces errors
- **Insufficient VRAM**: Use LoRA + quantization + gradient checkpointing

## Command Reference

| Command | Purpose |
|---------|---------|
| `llamafactory-cli train <config.yaml>` | Train or fine-tune model |
| `llamafactory-cli eval <config.yaml>` | Evaluate model on benchmarks |
| `llamafactory-cli export <config.yaml>` | Merge LoRA or quantize model |
| `llamafactory-cli chat <config.yaml>` | Interactive command-line chat |
| `llamafactory-cli webchat <config.yaml>` | Web-based chat interface |
| `llamafactory-cli api <config.yaml>` | Deploy OpenAI-compatible API |
| `llamafactory-cli webui` | Launch zero-code training WebUI |
| `llamafactory-cli version` | Check installation version |
| `llamafactory-cli env` | Verify environment setup |

## Quick Decision Guide

**Choose your training stage:**
- Want to adapt to instructions? → **SFT**
- Want to align with preferences? → **DPO** or **KTO**
- Want RLHF with reward model? → **PPO**
- Want to pre-train from scratch? → **PT**

**Choose your fine-tuning method:**
- Limited GPU (< 24GB)? → **LoRA**
- Moderate GPU (24-48GB)? → **Freeze** or **LoRA**
- High GPU (> 48GB)? → **Full** parameter

**Choose your acceleration:**
- Modern NVIDIA GPU (A100, H100, 4090)? → **FlashAttention-2**
- Want fastest training? → **Unsloth** + **FlashAttention-2**
- Memory constrained? → **Gradient checkpointing** + **LoRA** + **4-bit**

**Choose your deployment:**
- Quick testing? → **CLI chat** or **WebUI**
- Production API? → **vLLM** backend
- Edge/mobile? → **Quantize** to 4-bit

## Resources

- Official Documentation: https://github.com/hiyouga/LLaMA-Factory
- Supported Models: Check `model_name` in config
- Example Configs: `/examples/` directory in repository
- Community: GitHub Issues and Discussions

---

**Note**: This skill provides general guidance. Always consult the official LLaMA-Factory documentation for the most up-to-date information and model-specific requirements.
