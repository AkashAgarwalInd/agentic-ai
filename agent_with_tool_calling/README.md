# agent_with_tool_calling

A LangGraph agent that demonstrates tool calling with Gemini LLM and gold price fetching, including memory and SQLite checkpoint persistence.

## Overview

This agent shows how to:

- Use `ChatGoogleGenerativeAI` with tool binding (`bind_tools`)
- Define custom tools (`get_gold_price`)
- Use conditional edges based on tool call decisions
- Manage message state with `add_messages`
- **Persist state across restarts using SQLite checkpointers**
- **Maintain conversational memory using thread_id configuration**

## Setup

```bash
# Install dependencies
uv sync

# Set your Google API key
export GOOGLE_API_KEY=your_api_key_here

# Gold API key is loaded from .env file
cat agent_with_tool_calling/.env
```

## Usage

```bash
# Run the agent with memory and checkpointer
uv run python3 agent_with_tool_calling/agent_with_tool_calling.py
```

The agent supports two modes:

### With Checkpointer (Persistence)
State is saved to SQLite database (`checkpoints.db`) and persists across agent restarts.
Thread IDs allow multiple independent conversations.

```bash
uv run python3 agent_with_tool_calling/agent_with_tool_calling.py
```

### Without Checkpointer
State is kept in memory only and lost when the agent stops.

## Features

### SQLite Checkpointer Persistence

Checkpoints are saved to a SQLite database, enabling:

- **Cross-session persistence**: State survives agent restarts
- **Thread isolation**: Multiple conversations via unique `thread_id` values
- **Queryable history**: Checkpoints can be inspected via SQL

```python
from agent_with_tool_calling.agent_with_tool_calling import create_graph, list_checkpoints, list_thread_ids

# Create graph with persistence
graph = create_graph(use_checkpointer=True, db_path="my_checkpoints.db")

# List all thread IDs in the database
thread_ids = list_thread_ids("my_checkpoints.db")
# Returns: ["session_1", "session_2", ...]

# List all checkpoints
checkpoints = list_checkpoints("my_checkpoints.db")
```

### Conversational Memory

Memory is managed through the `thread_id` configuration parameter. When the same `thread_id` is used across multiple `invoke()` calls, the conversation history is preserved.

```python
config = {"configurable": {"thread_id": "user_123"}}

# First message
result1 = graph.invoke({"messages": [HumanMessage(content="Hi")]}, config)

# Second message - memory from first is preserved
result2 = graph.invoke({"messages": [HumanMessage(content="Remember I like gold?")]}, config)
```

## Project Structure

- `agent_with_tool_calling.py` - Main agent with tool calling, checkpointer, and memory
- `pyproject.toml` - Project dependencies
- `.env` - Contains `GOLD_API_KEY=goldapi-04af8e489a9ee617b72d0d5880d6bbc3-io`
- `README.md` - This file
- `llm_context.md` - Context for the LLM
- `checkpoints.db` - SQLite database for persistence (created on first run with checkpointer)