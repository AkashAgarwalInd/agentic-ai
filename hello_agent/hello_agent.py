from typing import TypedDict, Annotated
import os
from langgraph.graph import StateGraph, add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


class State(TypedDict):
    messages: Annotated[list, add_messages]


def think(state: State):
    last_message = state["messages"][-1]
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=os.getenv("GOOGLE_API_KEY"))
    response = llm.invoke([last_message])
    return {"messages": [AIMessage(content=response.content)]}


def create_graph():
    graph = StateGraph(State)
    graph.add_node("think", think)
    graph.add_edge("think", "__end__")
    graph.set_entry_point("think")
    return graph.compile()


if __name__ == "__main__":
    graph = create_graph()
    result = graph.invoke({"messages": [HumanMessage(content="Hello!")]})
    print(result["messages"][-1].content)