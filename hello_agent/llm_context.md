# Context for LLM - hello_agent

## Project: hello_agent
A simple LangGraph agent using Google's Gemini LLM (gemini-1.5-flash) to process messages.

## Agent Workflow

1. **Entry**: Graph starts at `think` node
2. **Processing**: `think` function:
   - Retrieves the last message from state
   - Calls `ChatGoogleGenerativeAI(model="gemini-1.5-flash")`
   - Invokes LLM with the last message
   - Returns AIMessage response appended to state
3. **Exit**: Graph routes to `__end__`

## State Structure

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
```

## Key Components

- **think(state)**: Core function that processes messages and calls Gemini
- **StateGraph**: LangGraph workflow with single node "think"
- **add_messages**: Utility function for appending messages to state
- **ChatGoogleGenerativeAI**: Gemini LLM integration

## Usage

```bash
export GOOGLE_API_KEY=your_key_here
uv run python3 hello_agent.py
```

Output: AI response appended to conversation state, e.g., "Thinking about: Hello!"