---
name: langchain-skills
description: Expert guidance for building applications with LangChain, LangGraph, and LangSmith SDKs. Provides code examples, best practices, and implementation patterns for Python and JavaScript/TypeScript based on official documentation.
---

# LangChain Skills

Expert guidance for building applications using the LangChain ecosystem: LangChain for LLM application development, LangGraph for stateful agent workflows, and LangSmith for tracing and evaluation.

## Quick Start

### Build a Basic Agent

Start with a simple agent that can answer questions and call tools:

**Python:**
```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

**JavaScript/TypeScript:**
```javascript
import { createAgent, tool } from "langchain";
import * as z from "zod";

const getWeather = tool(
  (input) => `It's always sunny in ${input.city}!`,
  {
    name: "get_weather",
    description: "Get the weather for a given city",
    schema: z.object({
      city: z.string().describe("The city to get the weather for"),
    }),
  }
);

const agent = createAgent({
  model: "claude-sonnet-4-5-20250929",
  tools: [getWeather],
});

await agent.invoke({
  messages: [{ role: "user", content: "What's the weather in Tokyo?" }],
});
```

## Core Concepts

### Agents

Agents combine language models with tools to create systems that can reason about tasks, decide which tools to use, and iteratively work towards solutions.

**Agent Loop (ReAct Pattern):**
```
User Input → Model (reasoning) → Tool Call → Tool Result → Model → Final Answer
```

**Key Components:**
- **Model**: The reasoning engine (e.g., GPT-4, Claude)
- **Tools**: Functions the agent can call
- **System Prompt**: Defines agent behavior
- **Memory**: Conversation state across turns

### Tools

Tools are components that agents call to perform actions. They extend model capabilities by letting them interact with external systems.

**Basic Tool Definition (Python):**
```python
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"
```

**Basic Tool Definition (JavaScript):**
```javascript
import * as z from "zod";
import { tool } from "langchain";

const searchDatabase = tool(
  ({ query, limit }) => `Found ${limit} results for '${query}'`,
  {
    name: "search_database",
    description: "Search the customer database for records matching the query.",
    schema: z.object({
      query: z.string().describe("Search terms to look for"),
      limit: z.number().describe("Maximum number of results to return"),
    }),
  }
);
```

**Type hints are required** as they define the tool's input schema. The docstring should be informative to help the model understand when to use the tool.

### Tool Runtime Context

Tools can access runtime information for context-aware behavior:

**Python:**
```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"
```

**JavaScript:**
```javascript
import { type Runtime } from "@langchain/langgraph";
import { tool } from "langchain";
import * as z from "zod";

type AgentRuntime = Runtime<{ user_id: string }>;

const getUserLocation = tool(
  (_, config: AgentRuntime) => {
    const { user_id } = config.context;
    return user_id === "1" ? "Florida" : "SF";
  },
  {
    name: "get_user_location",
    description: "Retrieve user information based on user ID",
  }
);
```

## Production-Ready Agent Example

This example demonstrates key production concepts:

**Python:**
```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

# Define system prompt
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:
- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location."""

# Define context schema
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

# Define tools
@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

# Configure model
model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)

# Define response format
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    punny_response: str
    weather_conditions: str | None = None

# Set up memory
checkpointer = InMemorySaver()

# Create agent
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

# Run agent
config = {"configurable": {"thread_id": "1"}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="Florida is still having a 'sun-derful' day!...",
#     weather_conditions="It's always sunny in Florida!"
# )
```

**JavaScript:**
```javascript
import { createAgent, tool, initChatModel } from "langchain";
import { MemorySaver, type Runtime } from "@langchain/langgraph";
import * as z from "zod";

// Define system prompt
const systemPrompt = `You are an expert weather forecaster, who speaks in puns.

You have access to two tools:
- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location`;

// Define tools
const getWeather = tool(
  ({ city }) => `It's always sunny in ${city}!`,
  {
    name: "get_weather_for_location",
    description: "Get the weather for a given city",
    schema: z.object({
      city: z.string(),
    }),
  }
);

const getUserLocation = tool(
  (_, config: Runtime<{ user_id: string}>) => {
    const { user_id } = config.context;
    return user_id === "1" ? "Florida" : "SF";
  },
  {
    name: "get_user_location",
    description: "Retrieve user information based on user ID",
    schema: z.object({}),
  }
);

// Configure model
const model = await initChatModel(
  "claude-sonnet-4-5-20250929",
  { temperature: 0.5, timeout: 10, maxTokens: 1000 }
);

// Define response format
const responseFormat = z.object({
  punny_response: z.string(),
  weather_conditions: z.string().optional(),
});

// Set up memory
const checkpointer = new MemorySaver();

// Create agent
const agent = createAgent({
  model: "claude-sonnet-4-5-20250929",
  systemPrompt: systemPrompt,
  tools: [getUserLocation, getWeather],
  responseFormat,
  checkpointer,
});

// Run agent
const config = {
  configurable: { thread_id: "1" },
  context: { user_id: "1" },
};

const response = await agent.invoke(
  { messages: [{ role: "user", content: "what is the weather outside?" }] },
  config
);
console.log(response.structuredResponse);
```

## RAG (Retrieval-Augmented Generation)

Build RAG applications to answer questions about your documents.

### RAG Agent Implementation

**Python:**
```python
from langchain.tools import tool
from langchain.agents import create_agent

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_context]
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)

# Use the agent
query = "What is task decomposition?"
for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
```

**JavaScript:**
```javascript
import { createAgent, tool } from "langchain";
import * as z from "zod";

const retrieve = tool(
  async ({ query }) => {
    const retrievedDocs = await vectorStore.similaritySearch(query, 2);
    const serialized = retrievedDocs
      .map(
        (doc) => `Source: ${doc.metadata.source}\nContent: ${doc.pageContent}`
      )
      .join("\n");
    return [serialized, retrievedDocs];
  },
  {
    name: "retrieve",
    description: "Retrieve information related to a query.",
    schema: z.object({ query: z.string() }),
    responseFormat: "content_and_artifact",
  }
);

const agent = createAgent({
  model: "gpt-5",
  tools: [retrieve],
  systemPrompt: "You have access to a tool that retrieves context. Use it to help answer queries."
});
```

### Document Indexing

**1. Load Documents**

**Python:**
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
docs = loader.load()
```

**JavaScript:**
```javascript
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

const loader = new PDFLoader("document.pdf");
const docs = await loader.load();
```

**2. Split Documents**

**Python:**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)
```

**JavaScript:**
```javascript
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const allSplits = await splitter.splitDocuments(docs);
```

**3. Store in Vector Database**

**Python:**
```python
# Add documents to vector store
document_ids = vector_store.add_documents(documents=all_splits)
```

**JavaScript:**
```javascript
await vectorStore.addDocuments(allSplits);
```

**4. Retrieve Documents**

**Python:**
```python
# Similarity search
results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)

# With scores
results = vector_store.similarity_search_with_score("query")
doc, score = results[0]
```

**JavaScript:**
```javascript
// Similarity search
const results = await vectorStore.similaritySearch("query text", 2);

// With scores
const resultsWithScores = await vectorStore.similaritySearchWithScore("query");
```

## Streaming

Stream real-time updates from your agent to improve user experience.

### Stream Agent Progress

**Python:**
```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="updates",
):
    for step, data in chunk.items():
        print(f"step: {step}")
        print(f"content: {data['messages'][-1].content_blocks}")
```

**JavaScript:**
```javascript
for await (const chunk of await agent.stream(
    { messages: [{ role: "user", content: "what is the weather in sf" }] },
    { streamMode: "updates" }
)) {
    const [step, content] = Object.entries(chunk)[0];
    console.log(`step: ${step}`);
}
```

### Stream LLM Tokens

**Python:**
```python
for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="messages",
):
    print(f"node: {metadata['langgraph_node']}")
    print(f"content: {token.content_blocks}")
```

**JavaScript:**
```javascript
for await (const [token, metadata] of await agent.stream(
    { messages: [{ role: "user", content: "what is the weather in sf" }] },
    { streamMode: "messages" }
)) {
    console.log(`node: ${metadata.langgraph_node}`);
    console.log(`content: ${JSON.stringify(token.contentBlocks)}`);
}
```

### Stream Custom Updates

**Python:**
```python
from langgraph.config import get_stream_writer

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"

# Use stream_mode="custom" to see custom updates
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="custom"
):
    print(chunk)
```

**JavaScript:**
```javascript
const getWeather = tool(
    async (input, config) => {
        config.writer?.(`Looking up data for city: ${input.city}`);
        config.writer?.(`Acquired data for city: ${input.city}`);
        return `It's always sunny in ${input.city}!`;
    },
    {
        name: "get_weather",
        description: "Get weather for a given city.",
        schema: z.object({ city: z.string() }),
    }
);

for await (const chunk of await agent.stream(
    { messages: [{ role: "user", content: "what is the weather in sf" }] },
    { streamMode: "custom" }
)) {
    console.log(chunk);
}
```

## Models

Initialize and configure language models for your agents.

### Initialize a Model

**Python:**
```python
from langchain.chat_models import init_chat_model

# Simple initialization
model = init_chat_model("claude-sonnet-4-5-20250929")

# With parameters
model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
)
```

**JavaScript:**
```javascript
import { initChatModel } from "langchain";

// Simple initialization
const model = await initChatModel("claude-sonnet-4-5-20250929");

// With parameters
const model = await initChatModel(
    "claude-sonnet-4-5-20250929",
    { temperature: 0.7, timeout: 30, maxTokens: 1000 }
);
```

### Use Model Directly

**Python:**
```python
# Single message
response = model.invoke("Why do parrots talk?")

# Conversation
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Translate: I love programming."},
    {"role": "assistant", "content": "J'adore la programmation."},
    {"role": "user", "content": "Translate: I love building applications."}
]
response = model.invoke(conversation)
```

**JavaScript:**
```javascript
// Single message
const response = await model.invoke("Why do parrots talk?");

// Conversation
const conversation = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", "Translate: I love programming." },
  { role: "assistant", content: "J'adore la programmation." },
  { role: "user", content: "Translate: I love building applications." },
];
const response = await model.invoke(conversation);
```

### Stream Model Output

**Python:**
```python
for chunk in model.stream("Why do parrots have colorful feathers?"):
    print(chunk.text, end="|", flush=True)
```

**JavaScript:**
```javascript
const stream = await model.stream("Why do parrots have colorful feathers?");
for await (const chunk of stream) {
  console.log(chunk.text);
}
```

## Structured Output

Get predictable, structured responses from your agent.

**Python:**
```python
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

# Using ToolStrategy (works with any model that supports tool calling)
agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)
)

# Using ProviderStrategy (uses native structured output)
agent = create_agent(
    model="gpt-4o",
    response_format=ProviderStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

**JavaScript:**
```javascript
import * as z from "zod";
import { createAgent } from "langchain";

const ContactInfo = z.object({
  name: z.string(),
  email: z.string(),
  phone: z.string(),
});

const agent = createAgent({
  model: "gpt-4o",
  responseFormat: ContactInfo,
});

const result = await agent.invoke({
  messages: [
    {
      role: "user",
      content: "Extract contact info from: John Doe, john@example.com, (555) 123-4567",
    },
  ],
});

console.log(result.structuredResponse);
// { name: 'John Doe', email: 'john@example.com', phone: '(555) 123-4567' }
```

## Memory and State

### Short-Term Memory (Within Session)

Use checkpointers to maintain conversation state:

**Python:**
```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=checkpointer
)

# Use thread_id to maintain conversation context
config = {"configurable": {"thread_id": "1"}}

response1 = agent.invoke(
    {"messages": [{"role": "user", "content": "My name is Alice"}]},
    config=config
)

# Agent remembers the context
response2 = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    config=config
)
```

**JavaScript:**
```javascript
import { MemorySaver } from "@langchain/langgraph";

const checkpointer = new MemorySaver();

const agent = createAgent({
  model,
  tools,
  checkpointer,
});

const config = { configurable: { thread_id: "1" } };

const response1 = await agent.invoke(
  { messages: [{ role: "user", content: "My name is Alice" }] },
  config
);

// Agent remembers the context
const response2 = await agent.invoke(
  { messages: [{ role: "user", content: "What's my name?" }] },
  config
);
```

### Long-Term Memory (Across Sessions)

Use stores for persistent memory:

**Python:**
```python
from langgraph.store.memory import InMemoryStore
from langchain.tools import tool, ToolRuntime

@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info."""
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

@tool
def save_user_info(user_id: str, user_info: dict, runtime: ToolRuntime) -> str:
    """Save user info."""
    store = runtime.store
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."

store = InMemoryStore()
agent = create_agent(
    model,
    tools=[get_user_info, save_user_info],
    store=store
)
```

**JavaScript:**
```javascript
import { InMemoryStore } from "@langchain/langgraph";
import { tool } from "langchain";
import * as z from "zod";

const store = new InMemoryStore();

const getUserInfo = tool(
  async ({ user_id }) => {
    const value = await store.get(["users"], user_id);
    return value || "Unknown user";
  },
  {
    name: "get_user_info",
    description: "Look up user info.",
    schema: z.object({ user_id: z.string() }),
  }
);

const saveUserInfo = tool(
  async ({ user_id, name, age, email }) => {
    await store.put(["users"], user_id, { name, age, email });
    return "Successfully saved user info.";
  },
  {
    name: "save_user_info",
    description: "Save user info.",
    schema: z.object({
      user_id: z.string(),
      name: z.string(),
      age: z.number(),
      email: z.string(),
    }),
  }
);

const agent = createAgent({
  model,
  tools: [getUserInfo, saveUserInfo],
  store,
});
```

## Best Practices

### Tool Design
- Use descriptive names and clear docstrings
- Include type hints for all parameters
- Keep tools focused on single responsibilities
- Handle errors gracefully within tools
- Use ToolRuntime for context-aware behavior

### Agent Configuration
- Start with clear, specific system prompts
- Set appropriate temperature (lower for deterministic, higher for creative)
- Configure timeouts to prevent hanging
- Use structured output for predictable responses
- Enable LangSmith tracing for debugging

### Production Deployment
- Use persistent checkpointers (not InMemorySaver)
- Implement proper error handling and retries
- Monitor token usage and costs
- Cache embeddings when possible
- Use async operations for better performance
- Add rate limiting for API calls

### RAG Implementation
- Choose appropriate chunk sizes (usually 500-1000 tokens)
- Use overlap between chunks (50-200 tokens)
- Optimize retrieval parameters (k, score_threshold)
- Consider hybrid search for better results
- Evaluate retrieval quality regularly

## Common Integration Patterns

### Vector Store Initialization

**Python:**
```python
# Chroma (local)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    docs,
    embedding=OpenAIEmbeddings()
)

# FAISS (in-memory)
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, embeddings)
```

**JavaScript:**
```javascript
// Chroma
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { OpenAIEmbeddings } from "@langchain/openai";

const vectorStore = await Chroma.fromDocuments(
  docs,
  new OpenAIEmbeddings()
);

// FAISS
import { FaissStore } from "@langchain/community/vectorstores/faiss";

const vectorStore = await FaissStore.fromDocuments(docs, embeddings);
```

### LangSmith Tracing

Enable tracing for debugging and monitoring:

**Python:**
```python
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-api-key"

# All agent executions are now traced automatically
```

**JavaScript:**
```javascript
process.env.LANGSMITH_TRACING = "true";
process.env.LANGSMITH_API_KEY = "your-api-key";

// All agent executions are now traced automatically
```

## Troubleshooting

**Import Errors**: Ensure correct package installation
- `langchain`: Main package
- `langchain-core`: Core abstractions
- `langchain-openai`: OpenAI integration
- `langchain-anthropic`: Anthropic integration
- `langchain-community`: Community integrations
- `langchain-text-splitters`: Text splitting utilities
- `langgraph`: Graph-based workflows

**Streaming Issues**: Ensure all components support streaming and use proper async patterns

**Memory Problems**: Use appropriate checkpointers for production (not InMemorySaver)

**Rate Limits**: Implement exponential backoff and batch operations

**Tool Execution Errors**: Check tool signatures, type hints, and error handling

## Additional Resources

This skill provides practical guidance for using LangChain SDKs based on official documentation. For the most up-to-date API references and detailed guides, consult the official LangChain documentation and any local documentation repository available in your environment.
