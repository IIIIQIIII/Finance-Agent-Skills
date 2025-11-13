---
name: claude-code-router-skills
description: Expert guidance for setting up, configuring, and troubleshooting Claude Code Router - a powerful tool for routing Claude Code requests to different LLM models and providers. Use when working with multi-model setups, cost optimization, provider switching, or advanced Claude Code configurations.
---

# Claude Code Router Expert Assistant

## Overview

Claude Code Router is a sophisticated proxy tool that routes Claude Code requests to different LLM models and providers based on context, cost optimization, and specific use cases. This skill provides expert guidance for setup, configuration, troubleshooting, and advanced usage patterns.

---

# Core Capabilities & Use Cases

## 🎯 When to Use Claude Code Router

**Cost Optimization:**
- Use cheaper/local models for background tasks
- Route expensive models only for complex reasoning
- Optimize token usage with model-specific routing

**Multi-Model Workflows:**
- Different models for different capabilities (coding vs reasoning)
- Provider redundancy and failover
- A/B testing different models

**Advanced Integrations:**
- GitHub Actions with custom models
- CI/CD pipeline optimization
- Custom transformer development

**Enterprise Scenarios:**
- Multiple API key management
- Team-specific model routing
- Compliance and governance requirements

---

# Setup & Configuration Guide

## 🚀 Installation Process

### Step 1: Prerequisites Check
First, verify Claude Code is installed and working:

```bash
# Check Claude Code installation
claude-code --version
npm list -g @anthropic-ai/claude-code

# Install if needed
npm install -g @anthropic-ai/claude-code
```

### Step 2: Install Claude Code Router
```bash
# Install globally
npm install -g @musistudio/claude-code-router

# Verify installation
ccr --version
ccr --help
```

### Step 3: Initial Configuration
```bash
# Create configuration directory
mkdir -p ~/.claude-code-router

# Start with UI mode for easy setup (recommended for beginners)
ccr ui

# Or use CLI model manager
ccr model
```

## ⚙️ Configuration Deep Dive

### Core Configuration Structure

Create `~/.claude-code-router/config.json` with these sections:

```json
{
  "LOG": true,
  "LOG_LEVEL": "debug",
  "API_TIMEOUT_MS": 600000,
  "NON_INTERACTIVE_MODE": false,
  "APIKEY": "your-secret-key",
  "HOST": "127.0.0.1",
  "PROXY_URL": "http://127.0.0.1:7890",

  "Providers": [
    // Provider configurations
  ],

  "Router": {
    "default": "provider,model",
    "background": "provider,model",
    "think": "provider,model",
    "longContext": "provider,model",
    "longContextThreshold": 60000,
    "webSearch": "provider,model",
    "image": "provider,model"
  },

  "CUSTOM_ROUTER_PATH": "/path/to/custom-router.js",
  "transformers": [
    // Custom transformer configurations
  ]
}
```

### Environment Variable Management

**Best Practice: Use environment variables for API keys**

```json
{
  "OPENAI_API_KEY": "$OPENAI_API_KEY",
  "DEEPSEEK_API_KEY": "${DEEPSEEK_API_KEY}",
  "Providers": [
    {
      "name": "openai",
      "api_base_url": "https://api.openai.com/v1/chat/completions",
      "api_key": "$OPENAI_API_KEY",
      "models": ["gpt-4", "gpt-4-turbo"]
    }
  ]
}
```

Set environment variables:
```bash
export OPENAI_API_KEY="sk-your-key"
export DEEPSEEK_API_KEY="sk-your-deepseek-key"
```

### Provider Configuration Patterns

#### High-Performance Setup (OpenRouter + Local)
```json
"Providers": [
  {
    "name": "openrouter",
    "api_base_url": "https://openrouter.ai/api/v1/chat/completions",
    "api_key": "$OPENROUTER_API_KEY",
    "models": [
      "claude-sonnet-4-5",
      "google/gemini-2.0-flash-exp",
      "claude-haiku-4-5"
    ],
    "transformer": { "use": ["openrouter"] }
  },
  {
    "name": "ollama",
    "api_base_url": "http://localhost:11434/v1/chat/completions",
    "api_key": "ollama",
    "models": ["qwen2.5-coder:latest", "llama3.2:latest"]
  }
]
```

#### Cost-Optimized Setup (DeepSeek + Gemini)
```json
"Providers": [
  {
    "name": "deepseek",
    "api_base_url": "https://api.deepseek.com/chat/completions",
    "api_key": "$DEEPSEEK_API_KEY",
    "models": ["deepseek-chat", "deepseek-reasoner"],
    "transformer": {
      "use": ["deepseek"],
      "deepseek-chat": { "use": ["tooluse"] }
    }
  },
  {
    "name": "gemini",
    "api_base_url": "https://generativelanguage.googleapis.com/v1beta/models/",
    "api_key": "$GEMINI_API_KEY",
    "models": ["gemini-2.0-flash-exp", "gemini-1.5-pro"],
    "transformer": { "use": ["gemini"] }
  }
]
```

### Router Strategy Patterns

#### Development-Optimized Routing
```json
"Router": {
  "default": "deepseek,deepseek-chat",
  "background": "ollama,qwen2.5-coder:latest",
  "think": "openrouter,claude-sonnet-4-5",
  "longContext": "gemini,gemini-1.5-pro",
  "longContextThreshold": 40000,
  "webSearch": "openrouter,perplexity/llama-3.1-sonar-large-128k-online"
}
```

#### Production-Grade Routing
```json
"Router": {
  "default": "openrouter,claude-sonnet-4-5",
  "background": "deepseek,deepseek-chat",
  "think": "openrouter,claude-sonnet-4-5",
  "longContext": "openrouter,google/gemini-2.0-flash-exp",
  "longContextThreshold": 60000,
  "webSearch": "openrouter,google/gemini-2.0-flash-exp",
  "image": "openrouter,claude-sonnet-4-5"
}
```

---

# Advanced Features & Customization

## 🔧 Custom Transformers

### Built-in Transformer Reference

**Essential Transformers:**
- `anthropic`: Direct Anthropic API compatibility
- `openrouter`: OpenRouter API with provider routing
- `deepseek`: DeepSeek API optimization
- `gemini`: Google Gemini API adaptation
- `tooluse`: Enhanced tool usage optimization
- `maxtoken`: Custom token limits
- `reasoning`: Process reasoning content fields
- `enhancetool`: Error-tolerant tool calling
- `cleancache`: Remove cache control fields

### Creating Custom Transformers

Create `/path/to/custom-transformer.js`:

```javascript
/**
 * Custom transformer for specialized API adaptation
 * @param {object} request - The incoming request
 * @param {object} config - Router configuration
 * @param {object} options - Transformer options
 */
module.exports = {
  name: "custom-transformer",

  async transformRequest(request, config, options = {}) {
    // Modify request before sending to provider
    const transformed = { ...request };

    // Example: Add custom headers
    transformed.headers = {
      ...transformed.headers,
      'X-Custom-Header': options.customValue || 'default'
    };

    return transformed;
  },

  async transformResponse(response, config, options = {}) {
    // Modify response before sending to Claude Code
    const transformed = { ...response };

    // Example: Add custom metadata
    if (transformed.choices?.[0]?.message) {
      transformed.choices[0].message.custom_metadata = {
        transformer: "custom-transformer",
        timestamp: new Date().toISOString()
      };
    }

    return transformed;
  }
};
```

Register in `config.json`:
```json
{
  "transformers": [
    {
      "path": "/path/to/custom-transformer.js",
      "options": {
        "customValue": "production-mode"
      }
    }
  ]
}
```

## 🛠️ Custom Routing Logic

### Advanced Router Implementation

Create `/path/to/custom-router.js`:

```javascript
/**
 * Advanced custom router with intelligent model selection
 * @param {object} req - Request object from Claude Code
 * @param {object} config - Application configuration
 * @returns {Promise<string|null>} - "provider,model" or null for default
 */
module.exports = async function intelligentRouter(req, config) {
  const { messages, tools } = req.body;

  // Get the latest user message
  const userMessage = messages
    .slice()
    .reverse()
    .find(m => m.role === "user")?.content || "";

  // Calculate context complexity
  const totalTokens = messages.reduce((sum, msg) =>
    sum + (msg.content?.length || 0), 0
  );

  // Route based on content analysis
  if (userMessage.toLowerCase().includes("explain") ||
      userMessage.toLowerCase().includes("analyze")) {
    // Use powerful model for explanations
    return "openrouter,claude-sonnet-4-5";
  }

  if (userMessage.toLowerCase().includes("quick") ||
      userMessage.toLowerCase().includes("simple")) {
    // Use fast model for simple tasks
    return "deepseek,deepseek-chat";
  }

  // Route based on tool usage
  if (tools && tools.length > 0) {
    // Complex tool usage needs reliable model
    return "openrouter,claude-sonnet-4-5";
  }

  // Route based on context length
  if (totalTokens > 50000) {
    return "gemini,gemini-1.5-pro";
  }

  // Time-based routing (cost optimization)
  const hour = new Date().getHours();
  if (hour >= 22 || hour <= 6) {
    // Use cheaper models during off-peak hours
    return "deepseek,deepseek-chat";
  }

  // Fallback to default router
  return null;
};
```

### Subagent-Specific Routing

For routing within specific subagents, include routing instructions at the beginning of prompts:

```
<CCR-SUBAGENT-MODEL>openrouter,anthropic/claude-3.5-sonnet</CCR-SUBAGENT-MODEL>
Please analyze this complex codebase architecture...
```

---

# Operational Excellence

## 🚀 Running & Management

### Starting and Managing the Service

```bash
# Start the router service
ccr start

# Check service status
ccr status

# Stop the service
ccr stop

# Restart after configuration changes
ccr restart

# Run Claude Code through router
ccr code "Your prompt here"

# Dynamic model switching
/model openrouter,anthropic/claude-3.5-sonnet
```

### UI Mode for Configuration Management

```bash
# Start web-based configuration UI
ccr ui

# Access at http://localhost:3456 (or configured port)
```

**UI Features:**
- Visual configuration editor
- Real-time validation
- Model testing interface
- Status line configuration
- Provider management

### CLI Model Management

```bash
# Interactive model management
ccr model

# Features:
# - View current configuration
# - Switch models for different router types
# - Add new models to existing providers
# - Create new provider configurations
# - Configure transformers
```

## 📊 Monitoring & Observability

### Logging Configuration

```json
{
  "LOG": true,
  "LOG_LEVEL": "debug"
}
```

**Log Levels:** `fatal`, `error`, `warn`, `info`, `debug`, `trace`

**Log Locations:**
- Server logs: `~/.claude-code-router/logs/ccr-*.log`
- Application logs: `~/.claude-code-router/claude-code-router.log`

### Status Line Integration (Beta)

Enable in UI for real-time monitoring:
- Current model information
- Request routing status
- Performance metrics
- Error indicators

## 🔒 Security & Authentication

### API Key Protection

```json
{
  "APIKEY": "your-secret-key",
  "HOST": "127.0.0.1"
}
```

**Authentication Methods:**
- `Authorization: Bearer your-secret-key`
- `x-api-key: your-secret-key`

### Network Security

```json
{
  "HOST": "127.0.0.1",  // Localhost only (secure)
  "HOST": "0.0.0.0",    // All interfaces (requires APIKEY)
  "PROXY_URL": "http://127.0.0.1:7890"  // Corporate proxy
}
```

---

# Integration Patterns

## 🤖 GitHub Actions Integration

### Basic Workflow Setup

`.github/workflows/claude.yaml`:
```yaml
name: Claude Code with Router

on:
  issue_comment:
    types: [created]

jobs:
  claude:
    if: contains(github.event.comment.body, '@claude')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Environment
        run: |
          curl -fsSL https://bun.sh/install | bash
          mkdir -p $HOME/.claude-code-router

      - name: Configure Router
        run: |
          cat << 'EOF' > $HOME/.claude-code-router/config.json
          {
            "LOG": true,
            "NON_INTERACTIVE_MODE": true,
            "Providers": [
              {
                "name": "deepseek",
                "api_base_url": "https://api.deepseek.com/chat/completions",
                "api_key": "${{ secrets.DEEPSEEK_API_KEY }}",
                "models": ["deepseek-chat"],
                "transformer": { "use": ["deepseek", "tooluse"] }
              }
            ],
            "Router": {
              "default": "deepseek,deepseek-chat"
            }
          }
          EOF
        shell: bash

      - name: Start Router
        run: |
          nohup ~/.bun/bin/bunx @musistudio/claude-code-router@latest start &
          sleep 5
        shell: bash

      - name: Run Claude Code
        uses: anthropics/claude-code-action@beta
        env:
          ANTHROPIC_BASE_URL: http://localhost:3456
        with:
          anthropic_api_key: "router-proxy-token"
```

### Advanced CI/CD Patterns

```yaml
      - name: Multi-Stage Routing
        run: |
          cat << 'EOF' > $HOME/.claude-code-router/config.json
          {
            "NON_INTERACTIVE_MODE": true,
            "Providers": [
              {
                "name": "fast",
                "api_base_url": "https://api.deepseek.com/chat/completions",
                "api_key": "${{ secrets.DEEPSEEK_API_KEY }}",
                "models": ["deepseek-chat"],
                "transformer": { "use": ["deepseek"] }
              },
              {
                "name": "powerful",
                "api_base_url": "https://openrouter.ai/api/v1/chat/completions",
                "api_key": "${{ secrets.OPENROUTER_API_KEY }}",
                "models": ["anthropic/claude-3.5-sonnet"],
                "transformer": { "use": ["openrouter"] }
              }
            ],
            "Router": {
              "default": "fast,deepseek-chat",
              "think": "powerful,anthropic/claude-3.5-sonnet",
              "longContext": "powerful,anthropic/claude-3.5-sonnet"
            }
          }
          EOF
```

---

# Troubleshooting Guide

## 🐛 Common Issues & Solutions

### Installation Issues

**Issue: Global installation fails**
```bash
# Solution: Use alternative installation methods
npm install -g @musistudio/claude-code-router --force

# Or use npx for testing
npx @musistudio/claude-code-router start

# Or use specific version
npm install -g @musistudio/claude-code-router@1.0.40
```

**Issue: Command not found after installation**
```bash
# Check global npm bin path
npm bin -g
echo $PATH

# Add to PATH if needed
export PATH=$PATH:$(npm bin -g)
```

### Configuration Issues

**Issue: Invalid JSON configuration**
```bash
# Validate JSON syntax
cat ~/.claude-code-router/config.json | json_pp

# Use the UI for validation
ccr ui
```

**Issue: Environment variable interpolation not working**
```bash
# Check environment variables
echo $OPENAI_API_KEY

# Debug interpolation
ccr start --debug
```

**Issue: Provider authentication failures**
```bash
# Test API key manually
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"gpt-4","messages":[{"role":"user","content":"test"}]}' \
     https://api.openai.com/v1/chat/completions
```

### Service Issues

**Issue: Service won't start**
```bash
# Check port availability
lsof -i :3456

# Kill existing processes
pkill -f claude-code-router

# Start with debug logging
LOG_LEVEL=debug ccr start
```

**Issue: Connection refused**
```bash
# Check service status
ccr status

# Verify configuration
ccr ui

# Check logs
tail -f ~/.claude-code-router/logs/ccr-*.log
```

### Model Routing Issues

**Issue: Wrong model being used**
```bash
# Check current routing configuration
ccr model

# Verify in logs
grep "Routing to" ~/.claude-code-router/claude-code-router.log

# Test specific model
/model provider,model-name
```

**Issue: Model not found**
```bash
# List available models
ccr model

# Check provider configuration
cat ~/.claude-code-router/config.json | json_pp | grep -A 10 "models"
```

### Performance Issues

**Issue: Slow response times**
```bash
# Check API timeout settings
grep "API_TIMEOUT_MS" ~/.claude-code-router/config.json

# Monitor network latency
ping api.openrouter.ai

# Use faster models for background tasks
# Configure router with:
"background": "deepseek,deepseek-chat"
```

**Issue: Rate limiting**
```bash
# Check rate limit headers in logs
grep "rate" ~/.claude-code-router/logs/ccr-*.log

# Configure multiple providers for load balancing
# Implement custom router with fallback logic
```

## 🔍 Debug Modes & Diagnostic Tools

### Enable Debug Logging
```json
{
  "LOG": true,
  "LOG_LEVEL": "debug"
}
```

### Network Debugging
```bash
# Monitor HTTP traffic
sudo tcpdump -i any port 3456

# Check proxy configuration
curl -x http://127.0.0.1:7890 http://httpbin.org/ip
```

### Configuration Validation
```bash
# Use UI for real-time validation
ccr ui

# Test configuration without starting service
ccr config --validate
```

---

# Best Practices & Patterns

## 💡 Optimization Strategies

### Cost Optimization
1. **Tiered Routing**: Use cheaper models for simple tasks
2. **Background Tasks**: Route non-critical tasks to local models
3. **Time-based Routing**: Use expensive models only during peak productivity hours
4. **Token Management**: Set appropriate max_tokens limits

### Performance Optimization
1. **Provider Redundancy**: Configure multiple providers for the same capability
2. **Geographic Routing**: Use providers closer to your location
3. **Connection Pooling**: Reuse connections where possible
4. **Caching**: Implement response caching for repeated queries

### Security Best Practices
1. **Environment Variables**: Never hardcode API keys in configuration files
2. **Local Binding**: Bind to 127.0.0.1 unless remote access is needed
3. **API Key Rotation**: Regularly rotate API keys
4. **Access Control**: Use APIKEY for authentication in shared environments

### Configuration Management
1. **Version Control**: Keep configuration templates in version control (without secrets)
2. **Environment Separation**: Use different configurations for development/production
3. **Gradual Rollout**: Test new models in background routing first
4. **Documentation**: Document routing decisions and model capabilities

## 🎯 Advanced Use Cases

### Multi-tenant Setup
```json
{
  "CUSTOM_ROUTER_PATH": "/path/to/tenant-router.js",
  "Providers": [
    {
      "name": "tenant-a",
      "api_base_url": "https://tenant-a.openrouter.ai/api/v1/chat/completions",
      "api_key": "$TENANT_A_API_KEY",
      "models": ["anthropic/claude-3.5-sonnet"]
    },
    {
      "name": "tenant-b",
      "api_base_url": "https://tenant-b.openrouter.ai/api/v1/chat/completions",
      "api_key": "$TENANT_B_API_KEY",
      "models": ["anthropic/claude-3.5-sonnet"]
    }
  ]
}
```

### A/B Testing Framework
```javascript
// custom-ab-router.js
module.exports = async function abTestRouter(req, config) {
  const userId = req.headers['x-user-id'];
  const hash = require('crypto').createHash('md5').update(userId).digest('hex');
  const bucket = parseInt(hash.slice(0, 8), 16) % 100;

  if (bucket < 50) {
    return "provider-a,model-a";  // 50% traffic
  } else {
    return "provider-b,model-b";  // 50% traffic
  }
};
```

### Load Balancing
```javascript
// load-balancer-router.js
const providers = [
  "openrouter,anthropic/claude-3.5-sonnet",
  "deepseek,deepseek-chat",
  "gemini,gemini-1.5-pro"
];

let currentIndex = 0;

module.exports = async function loadBalancedRouter(req, config) {
  const selected = providers[currentIndex % providers.length];
  currentIndex++;
  return selected;
};
```

---

# Reference & Resources

## 📚 Configuration Templates

### Starter Template
```json
{
  "LOG": true,
  "API_TIMEOUT_MS": 300000,
  "Providers": [
    {
      "name": "openrouter",
      "api_base_url": "https://openrouter.ai/api/v1/chat/completions",
      "api_key": "$OPENROUTER_API_KEY",
      "models": ["anthropic/claude-3.5-sonnet"],
      "transformer": { "use": ["openrouter"] }
    }
  ],
  "Router": {
    "default": "openrouter,anthropic/claude-3.5-sonnet"
  }
}
```

### Production Template
```json
{
  "LOG": true,
  "LOG_LEVEL": "info",
  "API_TIMEOUT_MS": 600000,
  "APIKEY": "$CCR_API_KEY",
  "HOST": "127.0.0.1",
  "NON_INTERACTIVE_MODE": false,
  "Providers": [
    {
      "name": "primary",
      "api_base_url": "https://openrouter.ai/api/v1/chat/completions",
      "api_key": "$PRIMARY_API_KEY",
      "models": ["anthropic/claude-3.5-sonnet", "anthropic/claude-3-haiku"],
      "transformer": { "use": ["openrouter"] }
    },
    {
      "name": "fallback",
      "api_base_url": "https://api.deepseek.com/chat/completions",
      "api_key": "$FALLBACK_API_KEY",
      "models": ["deepseek-chat"],
      "transformer": { "use": ["deepseek"] }
    },
    {
      "name": "local",
      "api_base_url": "http://localhost:11434/v1/chat/completions",
      "api_key": "ollama",
      "models": ["qwen2.5-coder:latest"]
    }
  ],
  "Router": {
    "default": "primary,anthropic/claude-3.5-sonnet",
    "background": "local,qwen2.5-coder:latest",
    "think": "primary,anthropic/claude-3.5-sonnet",
    "longContext": "primary,anthropic/claude-3.5-sonnet",
    "longContextThreshold": 60000
  }
}
```

## 🔧 Command Reference

### Service Management
```bash
ccr start          # Start the router service
ccr stop           # Stop the router service
ccr restart        # Restart the service
ccr status         # Check service status
ccr --version      # Show version information
ccr --help         # Show help information
```

### Configuration & Models
```bash
ccr ui             # Open web-based configuration UI
ccr model          # Interactive model management CLI
ccr config         # Configuration management (if available)
```

### Claude Code Integration
```bash
ccr code "prompt"  # Run Claude Code through router
# Within Claude Code:
/model provider,model-name  # Switch models dynamically
```

## 🌐 Provider URLs Reference

| Provider | API Base URL | Transformer |
|----------|-------------|-------------|
| OpenRouter | `https://openrouter.ai/api/v1/chat/completions` | `openrouter` |
| DeepSeek | `https://api.deepseek.com/chat/completions` | `deepseek` |
| OpenAI | `https://api.openai.com/v1/chat/completions` | `anthropic` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/models/` | `gemini` |
| Groq | `https://api.groq.com/openai/v1/chat/completions` | `groq` |
| Ollama | `http://localhost:11434/v1/chat/completions` | none |
| SiliconFlow | `https://api.siliconflow.cn/v1/chat/completions` | none |
| Volcengine | `https://ark.cn-beijing.volces.com/api/v3/chat/completions` | `deepseek` |

## ⚡ Quick Actions

### Emergency Procedures
```bash
# Service not responding
pkill -f claude-code-router
ccr restart

# Reset configuration
cp ~/.claude-code-router/config.json ~/.claude-code-router/config.json.backup
ccr ui

# Check logs for errors
tail -f ~/.claude-code-router/logs/ccr-*.log
```

### Performance Tuning
```bash
# Reduce timeout for faster failover
# In config.json: "API_TIMEOUT_MS": 30000

# Enable compression for better performance
# Some providers support gzip compression

# Use local models for development
# "background": "ollama,qwen2.5-coder:latest"
```

---

*Remember: Always test configuration changes in a development environment before applying to production. Use `ccr ui` for safer configuration management, and monitor logs when troubleshooting issues.*