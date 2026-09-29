# agent_with_tool_calling

A LangGraph agent that demonstrates tool calling with Gemini LLM and gold price fetching.

## Overview

This agent shows how to:
- Use `ChatGoogleGenerativeAI` with tool binding (`bind_tools`)
- Define custom tools (`get_gold_price`)
- Use conditional edges based on tool call decisions
- Manage message state with `add_messages`

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
# Run the agent
uv run python3 agent_with_tool_calling/agent_with_tool_calling.py
```

```
export GOOGLE_API_KEY={} &&
export GOLD_API_KEY={} &&
uv run python3 agent_with_tool_calling/agent_with_tool_calling.py
```

The agent will:
1. Accept a user question about gold prices
2. Decide whether to call the `get_gold_price` tool
3. Execute the tool and return the gold price information
4. Print the final response

## Project Structure

- `agent_with_tool_calling.py` - Main agent with tool calling workflow
- `pyproject.toml` - Project dependencies
- `.env` - Contains `GOLD_API_KEY` for goldapi.io
- `README.md` - This file
- `llm_context.md` - Context for the LLM