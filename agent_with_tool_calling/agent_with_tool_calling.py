from typing import TypedDict, Annotated
import os
import sqlite3
import requests
from langgraph.graph import StateGraph, add_messages, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver


@tool
def get_gold_price() -> dict:
    """Fetch current gold price from goldapi.io"""
    api_key = os.getenv("GOLD_API_KEY")
    url = "https://www.goldapi.io/api/XAU/USD"
    headers = {
        "x-access-token": api_key,
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


class State(TypedDict):
    messages: Annotated[list, add_messages]


def think_node(state: State):
    """LLM node that decides whether to call tools"""
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=os.getenv("GOOGLE_API_KEY"))
    response = llm.bind_tools([get_gold_price]).invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: State):
    """Check last message for tool calls"""
    last_message = state["messages"][-1]
    # If the AI requested a tool call, continue to tool node
    if last_message.tool_calls:
        return "call_tool"
    # Otherwise end
    return "__end__"


def call_tool_node(state: State):
    """Execute the requested tool and return result"""
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    if tool_call["name"] == "get_gold_price":
        result = get_gold_price.invoke({})
        return {"messages": [AIMessage(content=f"Gold price info: {result}")]}
    return {"messages": [AIMessage(content="Unknown tool requested")]}


def create_graph(use_checkpointer: bool = True, db_path: str = "checkpoints.db", use_memory: bool = True):
    """Create graph with SQLite checkpointer for persistence.
    
    Args:
        use_checkpointer: If True, adds SqliteSaver for persistence across restarts
        db_path: Path to SQLite database file for checkpointer persistence
        use_memory: If True, adds MemorySaver for conversational memory (via thread_id config)
    
    Returns:
        Compiled StateGraph with checkpointer for persistence
    """
    # Initialize SQLite connection and checkpointer
    checkpointer = None
    if use_checkpointer:
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        conn = sqlite3.connect(db_path)
        checkpointer = SqliteSaver(conn)

    # Initialize memory saver (stores conversation history in the checkpointer store)
    memory = MemorySaver() if use_memory else None

    graph = StateGraph(State)
    graph.add_node("think", think_node)
    graph.add_node("call_tool", call_tool_node)
    graph.add_conditional_edges(
        "think",
        should_continue,
        {
            "call_tool": "call_tool",
            "__end__": "__end__",
        },
    )
    graph.add_edge("call_tool", "__end__")
    graph.set_entry_point("think")

    # Compile with checkpointer
    # Memory is handled through the checkpointer store and thread_id config during invoke
    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    compiled_graph = graph.compile(**compile_kwargs)

    # Return both the compiled graph and the memory saver if used
    if use_memory and memory:
        # Memory is automatically integrated through the checkpointer store
        # when thread_id is provided consistently during invoke
        return compiled_graph
    return compiled_graph


def invoke_with_memory(graph, messages, thread_id):
    """Invoke graph with memory and checkpointer support.
    
    Args:
        graph: Compiled StateGraph with checkpointer
        messages: List of messages to process
        thread_id: Unique thread identifier for checkpoint persistence and memory
    
    Returns:
        Graph invocation result with memory persisted
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }
    
    return graph.invoke({"messages": messages}, config)


def list_checkpoints(db_path="checkpoints.db"):
    """List existing checkpoints in SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM checkpoint")
        rows = cursor.fetchall()
        return rows
    finally:
        conn.close()


def list_thread_ids(db_path="checkpoints.db"):
    """List all thread IDs in the SQLite checkpoint store."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoint")
        rows = cursor.fetchall()
        return [row[0] for row in rows]
    finally:
        conn.close()


if __name__ == "__main__":
    # Example 1: Run with SQLite checkpointer
    print("=== Example 1: Graph with SQLite Checkpointer ===")
    graph = create_graph(use_checkpointer=True, db_path="agent_with_tool_calling/checkpoints.db", use_memory=True)

    # First interaction (thread_id: gold_price_session_1)
    config1 = {"configurable": {"thread_id": "gold_price_session_1"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content="What is the current gold price?")]},
        config1
    )
    print("First query - Last message:", result["messages"][-1].content[:100], "...")

    # Second interaction (same thread_id - checkpointer persists state)
    config1_again = {"configurable": {"thread_id": "gold_price_session_1"}}
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="Thank you")]},
        config1_again
    )
    print("Second query (same thread) - Last message:", result2["messages"][-1].content[:100], "...")

    # Example 2: Fresh session with different thread_id
    print("\n=== Example 2: Fresh Session (different thread_id) ===")
    config2 = {"configurable": {"thread_id": "gold_price_session_2"}}
    result3 = graph.invoke(
        {"messages": [HumanMessage(content="What is the current gold price?")]},
        config2
    )
    print("Fresh query - Last message:", result3["messages"][-1].content[:100], "...")

    # Example 3: Graph without checkpointer
    print("\n=== Example 3: Graph without Checkpointer ===")
    graph2 = create_graph(use_checkpointer=False, use_memory=False)
    result4 = graph2.invoke({"messages": [HumanMessage(content="What is the current gold price?")]})
    print("Fresh query - Last message:", result4["messages"][-1].content[:100], "...")

    # Example 4: List checkpoints in SQLite
    print("\n=== Example 4: SQLite Checkpoint Persistence ===")
    checkpoints = list_checkpoints("agent_with_tool_calling/checkpoints.db")
    print(f"Checkpoints found in DB: {len(checkpoints)}")

    # Example 5: List all thread IDs
    print("\n=== Example 5: Active Thread IDs ===")
    thread_ids = list_thread_ids("agent_with_tool_calling/checkpoints.db")
    print(f"Active thread IDs: {thread_ids}")