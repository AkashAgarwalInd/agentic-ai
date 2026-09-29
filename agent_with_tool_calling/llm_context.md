# Context for LLM - agent_with_tool_calling

## Project: agent_with_tool_calling
A LangGraph agent using Google's Gemini LLM (gemini-1.5-flash) with tool calling capabilities to fetch gold prices.

## Agent Workflow

1. **Entry**: Graph starts at `think` node
2. **Processing**: `think_node` function:
   - Receives conversation messages
   - Calls `ChatGoogleGenerativeAI(model="gemini-1.5-flash")` with bound tools
   - `get_gold_price` tool is available for the LLM to use
   - LLM decides whether to call a tool or finish
3. **Conditional Edge**: `should_continue` function checks if the last message has tool calls
   - If tool calls exist → routes to `call_tool` node
   - If no tool calls → routes to `__end__`
4. **Tool Execution**: `call_tool_node` function:
   - Extracts the tool call from the message
   - Executes `get_gold_price()` function
   - Returns the result as an AIMessage
5. **Exit**: Graph ends at `__end__`

## State Structure

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history with tool results
```

## Key Components

- **think_node(state)**: LLM node that binds `get_gold_price` tool and invokes with messages
- **should_continue(state)**: Conditional edge function - returns "call_tool" if tool calls present, "__end__" otherwise
- **call_tool_node(state)**: Executes the requested tool and returns result message
- **get_gold_price()**: Custom tool that fetches gold price from goldapi.io
- **StateGraph**: LangGraph workflow with conditional edges between think → call_tool → end
- **add_messages**: Utility function for appending messages to state

## Tool: get_gold_price

- **Purpose**: Fetch current gold price from goldapi.io
- **Implementation**: Uses `requests` library with API key from environment
- **Returns**: JSON response with gold spot price data or error information

## Usage

```bash
export GOOGLE_API_KEY=your_gemini_key_here
uv run python3 agent_with_tool_calling/agent_with_tool_calling.py
```

The GOLD_API_KEY is automatically loaded from the `.env` file in the `agent_with_tool_calling/` directory.

## Expected Behavior

- User asks about gold price → LLM may decide to call `get_gold_price` tool
- Tool executes and returns gold price data from goldapi.io
- Result is appended to conversation state
- Final response printed to console