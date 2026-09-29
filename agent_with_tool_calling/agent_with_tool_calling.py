from typing import TypedDict, Annotated
import os
import requests
from langgraph.graph import StateGraph, add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool


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


def create_graph():
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
    return graph.compile()


if __name__ == "__main__":
    graph = create_graph()
    result = graph.invoke({"messages": [HumanMessage(content="What is the current gold price?")]})
    print("Final messages:", result["messages"])
    if result["messages"]:
        print("Last message content:", result["messages"][-1].content)