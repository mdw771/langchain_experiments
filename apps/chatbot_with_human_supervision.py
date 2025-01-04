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
from langchain_core.messages import ToolMessage, BaseMessage, AIMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

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


def stream_graph_updates(user_input: str, config: dict = None, stream_mode: str = "values"):
    if user_input is None:
        input = None
    else:
        input = {"messages": [("user", user_input)]}
    events = graph.stream(input, config=config, stream_mode=stream_mode)
    for event in events:
        # Since we are using "values" for `stream_mode`, the event has a different structure
        # from previous examples (where they use the default "update" mode). Here, `event` is
        # no longer keyed by node names, but contains the "message" key directly.
        if isinstance(event["messages"], list):
            event["messages"][-1].pretty_print()
        else:
            event["messages"].pretty_print()


os.environ["OPENAI_API_KEY"] = util.get_api_key("openai")
os.environ["TAVILY_API_KEY"] = util.get_api_key("tavily")

tavily_search_tool = TavilySearchResults(max_results=2)
tools = [tavily_search_tool]

llm = ChatOpenAI(model="gpt-4o-mini")
llm = llm.bind_tools(tools)

memory = MemorySaver()

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
)
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"]
)
util.draw_graph(graph)

config = {"configurable": {"thread_id": "1"}}

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["q", "quit", "exit"]:
            print("Goodbye!")
            break
        elif user_input.lower() == "continue":
            stream_graph_updates(None, config=config, stream_mode="values")
        elif user_input.lower() == "override_ai":
            # Override AI response directly
            overriding_input = input("Overwrite input: ")
            existing_message = graph.get_state(config).values["messages"][-1]
            new_messages = [
                # Faked tool message
                ToolMessage(
                    content=overriding_input,
                    tool_call_id=existing_message.tool_calls[0]["id"]
                ),
                # Faked AI message
                AIMessage(content=overriding_input)
            ]
            new_messages[-1].pretty_print()
            graph.update_state(config, {"messages": new_messages}, as_node="chatbot")
            # Continue graph steaming
            stream_graph_updates(None, config=config, stream_mode="values")
        elif user_input.lower() == "override_tool":
            # Override search queries in the tool call
            overriding_input = input("Overwrite input: ")
            existing_message = graph.get_state(config).values["messages"][-1]
            new_tool_call = existing_message.tool_calls[0].copy()
            new_tool_call["args"]["query"] = overriding_input
            new_messages = [
                AIMessage(
                    content=existing_message.content,
                    tool_calls=[new_tool_call],
                    # Setting ID allows the existing message to be updated
                    id=existing_message.id),
            ]
            new_messages[-1].pretty_print()
            graph.update_state(config, {"messages": new_messages})
        else:
            stream_graph_updates(user_input, config=config, stream_mode="values")
    except IOError:
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input, config=config)
        break
