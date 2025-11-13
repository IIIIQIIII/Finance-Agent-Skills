---
name: claude-agent-sdk-python
description: Expert guide for building applications with the Claude Agent SDK for Python. Use when working with the SDK to create automated workflows, implement custom tools (in-process MCP servers), configure hooks for agent behavior control, or integrate Claude Code into Python applications.
---

# Claude Agent SDK for Python

Build Python applications that leverage Claude Code programmatically using the Claude Agent SDK. This skill helps you integrate Claude's capabilities into your Python workflows, create custom tools, and control agent behavior through hooks.

## What This Skill Does

This skill provides comprehensive guidance for:

1. **Basic queries** using the simple `query()` function for one-shot interactions
2. **Interactive sessions** with `ClaudeSDKClient` for bidirectional conversations
3. **Custom tools** implementation as in-process MCP servers (no subprocess overhead)
4. **Hooks** for deterministic processing and automated agent feedback
5. **Configuration** of tools, permissions, working directories, and system prompts
6. **Error handling** and best practices for production applications

## Prerequisites

Before using the SDK, ensure you have:

1. **Python 3.10+** installed
2. **Node.js** installed (required for Claude Code)
3. **Claude Code 2.0.0+** installed globally:
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```
4. **Claude Agent SDK** installed:
   ```bash
   pip install claude-agent-sdk
   ```

## Quick Start

### Basic Query Example

The simplest way to interact with Claude Code:

```python
import anyio
from claude_agent_sdk import query

async def main():
    async for message in query(prompt="What is 2 + 2?"):
        print(message)

anyio.run(main)
```

### Processing Responses

Extract text from assistant messages:

```python
from claude_agent_sdk import query, AssistantMessage, TextBlock

async def main():
    async for message in query(prompt="Explain Python in one sentence"):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Claude: {block.text}")

anyio.run(main)
```

## Core Components

### 1. The `query()` Function

**Purpose**: One-shot queries without maintaining conversation state

**Use when**: You need a simple question-answer interaction

**Parameters**:
- `prompt` (str): The question or task for Claude
- `options` (ClaudeAgentOptions, optional): Configuration options

**Returns**: `AsyncIterator` of messages

**Example with options**:

```python
from claude_agent_sdk import query, ClaudeAgentOptions

options = ClaudeAgentOptions(
    system_prompt="You are a Python expert that explains things simply",
    max_turns=1,
    allowed_tools=["Read", "Write", "Bash"]
)

async for message in query(
    prompt="Create a hello.py file",
    options=options
):
    print(message)
```

### 2. The `ClaudeSDKClient`

**Purpose**: Interactive, bidirectional conversations with state management

**Use when**: You need multiple rounds of conversation, custom tools, or hooks

**Key features**:
- Maintains conversation context
- Supports custom tools (in-process MCP servers)
- Enables hooks for behavior control
- Session management and forking

**Basic usage**:

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

options = ClaudeAgentOptions(
    system_prompt="You are a helpful assistant"
)

async with ClaudeSDKClient(options=options) as client:
    # Send a query
    await client.query("What is Python?")

    # Receive response
    async for msg in client.receive_response():
        print(msg)

    # Continue conversation
    await client.query("Tell me more about it")
    async for msg in client.receive_response():
        print(msg)
```

## Custom Tools (In-Process MCP Servers)

Custom tools are Python functions that Claude can invoke during conversations. They run in-process (no separate subprocess), making them faster and easier to debug than external MCP servers.

### Benefits Over External MCP Servers

✅ **No subprocess management** - Runs in the same process as your application
✅ **Better performance** - No IPC overhead for tool calls
✅ **Simpler deployment** - Single Python process instead of multiple
✅ **Easier debugging** - All code runs in the same process
✅ **Type safety** - Direct Python function calls with type hints

### Creating a Simple Tool

```python
from claude_agent_sdk import (
    tool,
    create_sdk_mcp_server,
    ClaudeAgentOptions,
    ClaudeSDKClient
)

# Define a tool using the @tool decorator
@tool("greet", "Greet a user by name", {"name": str})
async def greet_user(args):
    return {
        "content": [
            {"type": "text", "text": f"Hello, {args['name']}!"}
        ]
    }

# Create an SDK MCP server
server = create_sdk_mcp_server(
    name="my-tools",
    version="1.0.0",
    tools=[greet_user]
)

# Use it with Claude
options = ClaudeAgentOptions(
    mcp_servers={"tools": server},
    allowed_tools=["mcp__tools__greet"]
)

async with ClaudeSDKClient(options=options) as client:
    await client.query("Greet Alice")
    async for msg in client.receive_response():
        print(msg)
```

### Advanced Tool Example: Calculator

```python
from claude_agent_sdk import tool, create_sdk_mcp_server

@tool(
    "add",
    "Add two numbers together",
    {"a": float, "b": float}
)
async def add(args):
    result = args["a"] + args["b"]
    return {
        "content": [
            {"type": "text", "text": f"{args['a']} + {args['b']} = {result}"}
        ]
    }

@tool(
    "multiply",
    "Multiply two numbers",
    {"a": float, "b": float}
)
async def multiply(args):
    result = args["a"] * args["b"]
    return {
        "content": [
            {"type": "text", "text": f"{args['a']} × {args['b']} = {result}"}
        ]
    }

# Create calculator server
calculator = create_sdk_mcp_server(
    name="calculator",
    version="1.0.0",
    tools=[add, multiply]
)

options = ClaudeAgentOptions(
    mcp_servers={"calc": calculator},
    allowed_tools=["mcp__calc__add", "mcp__calc__multiply"]
)

async with ClaudeSDKClient(options=options) as client:
    await client.query("What is 15 times 23?")
    async for msg in client.receive_response():
        print(msg)
```

### Tool Naming Convention

When tools are registered, they use the pattern: `mcp__{server_name}__{tool_name}`

Example:
- Server: `"calculator"`
- Tool: `"add"`
- Full name: `"mcp__calculator__add"`

### Migration from External MCP Servers

**Before** (External MCP server - separate process):
```python
options = ClaudeAgentOptions(
    mcp_servers={
        "calculator": {
            "type": "stdio",
            "command": "python",
            "args": ["-m", "calculator_server"]
        }
    }
)
```

**After** (SDK MCP server - in-process):
```python
from my_tools import add, subtract

calculator = create_sdk_mcp_server(
    name="calculator",
    tools=[add, subtract]
)

options = ClaudeAgentOptions(
    mcp_servers={"calculator": calculator}
)
```

### Mixed Server Support

You can use both SDK and external MCP servers together:

```python
options = ClaudeAgentOptions(
    mcp_servers={
        "internal": sdk_server,      # In-process SDK server
        "external": {                # External subprocess server
            "type": "stdio",
            "command": "external-server"
        }
    }
)
```

## Hooks

Hooks are Python functions that Claude Code invokes at specific points in the agent loop. They enable deterministic processing and automated feedback without Claude's involvement.

### Available Hook Events

- `PreToolUse`: Before a tool is executed (can approve/deny)
- `PostToolUse`: After a tool completes
- `UserPromptSubmit`: When user submits a prompt
- Other events: See [Claude Code Hooks Reference](https://docs.anthropic.com/en/docs/claude-code/hooks)

### Basic Hook Example: Command Validation

Block dangerous bash commands before execution:

```python
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher
)

async def check_bash_command(input_data, tool_use_id, context):
    """Block commands containing forbidden patterns."""
    tool_name = input_data["tool_name"]
    tool_input = input_data["tool_input"]

    # Only check Bash commands
    if tool_name != "Bash":
        return {}

    command = tool_input.get("command", "")

    # Define forbidden patterns
    block_patterns = ["rm -rf /", "dd if=", ":(){ :|:& };:"]

    for pattern in block_patterns:
        if pattern in command:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Blocked: dangerous pattern '{pattern}' detected",
                }
            }

    # Allow command
    return {}

# Configure hook
options = ClaudeAgentOptions(
    allowed_tools=["Bash"],
    hooks={
        "PreToolUse": [
            HookMatcher(matcher="Bash", hooks=[check_bash_command]),
        ],
    }
)

async with ClaudeSDKClient(options=options) as client:
    # This will be blocked
    await client.query("Run: rm -rf /tmp/test")
    async for msg in client.receive_response():
        print(msg)

    # This will succeed
    await client.query("Run: echo 'Hello World'")
    async for msg in client.receive_response():
        print(msg)
```

### Hook Return Format

Hooks must return a dictionary. For `PreToolUse` hooks:

**To allow** (do nothing):
```python
return {}
```

**To deny**:
```python
return {
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": "Reason for blocking"
    }
}
```

**To approve**:
```python
return {
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "approve"
    }
}
```

### Advanced Hook Example: Logging and Metrics

```python
import json
from datetime import datetime

async def log_tool_usage(input_data, tool_use_id, context):
    """Log all tool usage for analytics."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "tool_name": input_data["tool_name"],
        "tool_use_id": tool_use_id,
        "input": input_data["tool_input"]
    }

    # Write to log file
    with open("tool_usage.log", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # Always allow (just logging)
    return {}

options = ClaudeAgentOptions(
    hooks={
        "PreToolUse": [
            HookMatcher(matcher="*", hooks=[log_tool_usage]),
        ]
    }
)
```

## Configuration Options

### ClaudeAgentOptions

Complete configuration reference:

```python
from pathlib import Path
from claude_agent_sdk import ClaudeAgentOptions

options = ClaudeAgentOptions(
    # System prompt
    system_prompt="You are a helpful Python expert",

    # Working directory
    cwd="/path/to/project",  # or Path("/path/to/project")

    # Tool configuration
    allowed_tools=["Read", "Write", "Bash", "Grep", "Glob"],

    # Permission mode
    permission_mode="acceptEdits",  # auto-accept file edits

    # Turn limits
    max_turns=10,  # max conversation turns

    # MCP servers
    mcp_servers={
        "custom": custom_sdk_server,
        "external": {
            "type": "stdio",
            "command": "server-command"
        }
    },

    # Hooks
    hooks={
        "PreToolUse": [
            HookMatcher(matcher="Bash", hooks=[bash_validator])
        ]
    },

    # Budget
    max_budget_usd=1.0  # spending limit
)
```

### Permission Modes

- `"ask"` (default): Prompt for each file edit
- `"acceptEdits"`: Auto-accept file edits (still prompts for other permissions)
- `"acceptAll"`: Auto-accept everything (use with caution)

### Working Directory

Set the working directory for Claude's operations:

```python
from pathlib import Path

# Using string path
options = ClaudeAgentOptions(cwd="/Users/alice/project")

# Using Path object
options = ClaudeAgentOptions(cwd=Path.home() / "project")
```

## Message Types

The SDK uses typed message objects for type safety:

### Message Type Hierarchy

```python
from claude_agent_sdk import (
    AssistantMessage,  # Claude's responses
    UserMessage,       # User inputs
    SystemMessage,     # System prompts
    ResultMessage,     # Final result with metadata
    TextBlock,         # Text content
    ToolUseBlock,      # Tool invocations
    ToolResultBlock    # Tool results
)
```

### Processing Different Message Types

```python
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock
)

async for message in query(prompt="What is Python?"):
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                print(f"Text: {block.text}")
            elif isinstance(block, ToolUseBlock):
                print(f"Tool: {block.name} with {block.input}")

    elif isinstance(message, ResultMessage):
        print(f"Cost: ${message.total_cost_usd:.4f}")
        print(f"Status: {message.status}")
```

## Error Handling

### Exception Hierarchy

```python
from claude_agent_sdk import (
    ClaudeSDKError,      # Base error
    CLINotFoundError,    # Claude Code not installed
    CLIConnectionError,  # Connection issues
    ProcessError,        # Process failed
    CLIJSONDecodeError,  # JSON parsing issues
)
```

### Comprehensive Error Handling

```python
from claude_agent_sdk import (
    query,
    CLINotFoundError,
    CLIConnectionError,
    ProcessError,
    CLIJSONDecodeError
)

async def safe_query(prompt: str):
    try:
        async for message in query(prompt=prompt):
            print(message)

    except CLINotFoundError:
        print("Error: Claude Code is not installed")
        print("Install with: npm install -g @anthropic-ai/claude-code")

    except CLIConnectionError as e:
        print(f"Connection error: {e}")
        print("Check if Claude Code is running properly")

    except ProcessError as e:
        print(f"Process failed with exit code: {e.exit_code}")
        print(f"Error output: {e.stderr}")

    except CLIJSONDecodeError as e:
        print(f"Failed to parse response: {e}")
        print(f"Raw output: {e.raw_output}")

    except Exception as e:
        print(f"Unexpected error: {e}")
```

## Available Tools

Common tools you can enable via `allowed_tools`:

- `"Read"` - Read files
- `"Write"` - Write/create files
- `"Edit"` - Edit existing files
- `"Bash"` - Execute bash commands
- `"Grep"` - Search file contents
- `"Glob"` - Find files by pattern
- `"WebFetch"` - Fetch web content
- `"WebSearch"` - Search the web
- `"Task"` - Launch specialized agents

Custom MCP tools: `"mcp__{server}__{tool}"`

See [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude) for complete list.

## Real-World Examples

### Example 1: Code Review Assistant

```python
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

async def code_review_assistant():
    """Interactive code review assistant."""
    options = ClaudeAgentOptions(
        system_prompt="""You are an expert code reviewer.
        Analyze code for bugs, security issues, and best practices.
        Provide constructive, actionable feedback.""",
        allowed_tools=["Read", "Grep", "Glob"],
        cwd="/path/to/project"
    )

    async with ClaudeSDKClient(options=options) as client:
        # Review all Python files
        await client.query(
            "Review all Python files in src/ for security vulnerabilities"
        )
        async for msg in client.receive_response():
            print(msg)

        # Follow-up question
        await client.query("Show me the most critical issues first")
        async for msg in client.receive_response():
            print(msg)
```

### Example 2: Data Processing Pipeline

```python
from claude_agent_sdk import tool, create_sdk_mcp_server

@tool("load_csv", "Load CSV data", {"filepath": str})
async def load_csv(args):
    import pandas as pd
    df = pd.read_csv(args["filepath"])
    return {
        "content": [{
            "type": "text",
            "text": f"Loaded {len(df)} rows, columns: {', '.join(df.columns)}"
        }]
    }

@tool("analyze_column", "Get statistics for a column",
      {"column": str})
async def analyze_column(args):
    # Assume df is accessible (use proper state management)
    stats = df[args["column"]].describe().to_dict()
    return {
        "content": [{
            "type": "text",
            "text": f"Statistics: {stats}"
        }]
    }

server = create_sdk_mcp_server(
    name="data",
    tools=[load_csv, analyze_column]
)

options = ClaudeAgentOptions(
    mcp_servers={"data": server},
    allowed_tools=["mcp__data__load_csv", "mcp__data__analyze_column"]
)

async with ClaudeSDKClient(options=options) as client:
    await client.query("Load data/sales.csv and analyze the revenue column")
    async for msg in client.receive_response():
        print(msg)
```

### Example 3: Automated Testing Agent

```python
async def test_runner():
    """Run tests and report failures."""
    options = ClaudeAgentOptions(
        system_prompt="You are a test automation expert",
        allowed_tools=["Bash", "Read", "Grep"],
        cwd="/path/to/project"
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("""
        Run pytest with coverage, then:
        1. Report failing tests
        2. Suggest fixes for each failure
        3. Check if related code needs tests
        """)

        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        print(block.text)
```

## Best Practices

### 1. Choose the Right API

**Use `query()` when**:
- Simple one-shot questions
- No conversation context needed
- No custom tools or hooks required

**Use `ClaudeSDKClient` when**:
- Multi-turn conversations
- Custom tools needed
- Hooks for behavior control
- Session management required

### 2. Tool Design

**Good tool design**:
```python
@tool(
    "search_users",
    "Search users by email or name. Returns user ID, name, and email.",
    {"query": str, "limit": int}
)
async def search_users(args):
    # Clear, focused purpose
    # Well-documented parameters
    # Reasonable scope
    pass
```

**Avoid**:
```python
@tool("do_everything", "Does stuff", {"data": dict})
async def do_everything(args):
    # Too vague
    # Unclear purpose
    # Overly broad scope
    pass
```

### 3. Error Handling

Always handle errors gracefully:

```python
try:
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for msg in client.receive_response():
            process(msg)
except ClaudeSDKError as e:
    log_error(e)
    notify_user(e)
```

### 4. Security

**For hooks validating commands**:
```python
# Use allowlist approach
ALLOWED_COMMANDS = {"ls", "echo", "cat", "grep"}

async def validate_command(input_data, tool_use_id, context):
    command = input_data["tool_input"]["command"].split()[0]
    if command not in ALLOWED_COMMANDS:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Command '{command}' not allowed"
            }
        }
    return {}
```

### 5. Resource Management

Always use context managers:

```python
# Good
async with ClaudeSDKClient(options=options) as client:
    await client.query(prompt)
    async for msg in client.receive_response():
        print(msg)

# Avoid manual cleanup
client = ClaudeSDKClient(options=options)
await client.connect()
# ... might not cleanup on error
await client.close()
```

### 6. Type Safety

Use type hints for better IDE support:

```python
from claude_agent_sdk import AssistantMessage, TextBlock

async def process_response(message: AssistantMessage) -> str:
    texts: list[str] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            texts.append(block.text)
    return " ".join(texts)
```

## Troubleshooting

### "CLINotFoundError: Claude Code not found"

**Problem**: Claude Code not installed or not in PATH

**Solution**:
```bash
npm install -g @anthropic-ai/claude-code
```

### "Connection refused" or "ProcessError"

**Problem**: Claude Code process failed to start

**Solutions**:
1. Check Claude Code version: `claude-code --version`
2. Verify Node.js installation: `node --version`
3. Check logs for errors
4. Try running Claude Code directly: `claude-code`

### Tools Not Working

**Problem**: Custom tools not being called

**Solutions**:
1. Verify tool is in `allowed_tools` list with correct name
2. Check tool name format: `mcp__{server}__{tool}`
3. Ensure server is in `mcp_servers` dict
4. Add logging to tool function to verify it's registered

### Hooks Not Triggering

**Problem**: Hook functions not being called

**Solutions**:
1. Verify hook event name matches (e.g., "PreToolUse")
2. Check matcher pattern matches tool name
3. Ensure hook returns correct format
4. Add print statements to debug

### High Costs

**Problem**: Unexpected API costs

**Solutions**:
1. Set `max_budget_usd` limit
2. Use `max_turns` to limit conversation length
3. Monitor `ResultMessage.total_cost_usd`
4. Limit allowed tools to only necessary ones

## Related Resources

- **Official Documentation**: [Claude Agent SDK Docs](https://docs.anthropic.com/en/docs/claude-code/sdk/sdk-python)
- **Claude Code Hooks**: [Hooks Reference](https://docs.anthropic.com/en/docs/claude-code/hooks)
- **GitHub Repository**: [claude-agent-sdk-python](https://github.com/anthropics/claude-agent-sdk-python)
- **Example Code**: See `examples/` directory in the repository
- **Type Definitions**: See `src/claude_agent_sdk/types.py`

## When to Use This Skill

Use this skill when you need to:

- **Integrate Claude Code** into Python applications programmatically
- **Build automated workflows** that leverage Claude's capabilities
- **Create custom tools** for Claude to use in your domain
- **Control agent behavior** through hooks and validation
- **Develop interactive applications** with multi-turn conversations
- **Process files and data** with Claude's assistance
- **Automate code reviews**, testing, documentation, or refactoring
- **Build domain-specific agents** with specialized tools and knowledge

This skill is essential for developers building production applications on top of Claude Code.
