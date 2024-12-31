import os
import sys
import json
sys.path.insert(0, os.path.join(os.getcwd(), os.pardir))

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage

import util


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    # Notice how the chatbot node function takes the current State 
    # as input and returns a dictionary containing an updated 
    # messages list under the key "messages". This is the basic 
    # pattern for all LangGraph node functions.
    ai_message = llm.invoke(state["messages"])
    return {"messages": ai_message}


class BasicToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict) -> dict:
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input.")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                )
            )
        return {"messages": outputs}


def route_tools(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        # `stream` returns a generator of events, where each event is a dictionary
        # containing the updated state of a certain node. 
        # Here, the first event is the updated state of the "chatbot" node.
        # The second event is the updated state of the "tools" node.
        # The third event is the updated state of the "chatbot" node again.
        for value in event.values():
            if isinstance(value["messages"], list):
                print("Assistant: ", value["messages"][-1].content)
            else:
                print("Assistant: ", value["messages"].content)


os.environ["OPENAI_API_KEY"] = util.get_api_key("openai")
os.environ["TAVILY_API_KEY"] = util.get_api_key("tavily")

tavily_search_tool = TavilySearchResults(max_results=2)
tools = [tavily_search_tool]

llm = ChatOpenAI(model="gpt-4o-mini")
llm = llm.bind_tools(tools)

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_node("tools", BasicToolNode(tools=tools))
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    path_map={"tools": "tools", END: END}
)
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile()
util.draw_graph(graph)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["q", "quit", "exit"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except IOError:
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
