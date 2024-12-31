import os
import sys
import json
sys.path.insert(0, os.path.join(os.getcwd(), os.pardir))

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

import util


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    # Notice how the chatbot node function takes the current State 
    # as input and returns a dictionary containing an updated 
    # messages list under the key "messages". This is the basic 
    # pattern for all LangGraph node functions.
    return {"messages": llm.invoke(state["messages"])}


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant: ", value["messages"].content)


os.environ["OPENAI_API_KEY"] = util.get_api_key()

llm = ChatOpenAI(model="gpt-3.5-turbo")

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["q", "quit", "exit"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
