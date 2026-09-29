# hello_agent

A simple LangGraph agent that uses Gemini LLM to respond to messages.

## Overview

This agent demonstrates a basic LangGraph workflow with:
- A `State` TypedDict using `Annotated[list, add_messages]` for message tracking
- A `think` node that calls `ChatGoogleGenerativeAI` (Gemini) to generate responses
- A StateGraph that processes messages through the think node

## Setup

```bash
# Install dependencies
uv sync

# Set your Google API key
export GOOGLE_API_KEY=your_api_key_here
```

## Usage

```bash
# Run the agent
uv run python3 hello_agent.py
```

```
export GOOGLE_API_KEY={} && uv run python3 hello_agent.py
```

## Expected Output

```
Thinking about: Hello!
```

The agent will:
1. Accept a `HumanMessage` as input
2. Call the Gemini LLM (`gemini-1.5-flash`) to generate a response
3. Append the AI response to the message state
4. Print the generated response

## Project Structure

- `hello_agent.py` - Main agent script with LangGraph workflow
- `pyproject.toml` - Project dependencies (langgraph, langchain-gemini)
- `README.md` - This file
- `uv.lock` - Lock file for uv package manager