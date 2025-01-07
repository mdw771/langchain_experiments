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
from pydantic import BaseModel

import util


class State(TypedDict):
    messages: Annotated[list, add_messages]
    ask_human: bool


class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.

    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """
    request: str


def chatbot(state: State):
    response = llm.invoke(state["messages"])
    ask_human = False
    if (
        response.tool_calls
        and response.tool_calls[0]["name"] == RequestAssistance.__name__
    ):
        ask_human = True
    return {"messages": [response], "ask_human": ask_human}


def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"]
    )


def human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        new_messages.append(
            create_response("No response from human.", state["messages"][-1])
        )
    return {
        "messages": new_messages,
        "ask_human": False
    }


def select_next_node(state: State):
    if state["ask_human"]:
        return "human"
    return tools_condition(state)


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
tools = [tavily_search_tool, RequestAssistance]

llm = ChatOpenAI(model="gpt-4o-mini")
llm = llm.bind_tools(tools)

memory = MemorySaver()

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_node("tools", ToolNode(tools=[tavily_search_tool]))
graph_builder.add_node("human", human_node)
graph_builder.add_conditional_edges(
    "chatbot",
    select_next_node,
    {"human": "human", "tools": "tools", END: END}
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["human"]
)
util.draw_graph(graph)

config = {"configurable": {"thread_id": "1"}}

while True:
    try:
        if len(graph.get_state(config).values) > 0 and graph.get_state(config).values["ask_human"]:
            human_response = input("Expert: ")
            ai_message = graph.get_state(config).values["messages"][-1]
            tool_message = create_response(human_response, ai_message)
            graph.update_state(config, {"messages": [tool_message]})
            stream_graph_updates(None, config=config, stream_mode="values")
        else:
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
            elif user_input.lower() == "override_tool_call":
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
            elif user_input.lower() == "replay":
                for id, state in enumerate(graph.get_state_history(config)):
                    print(f"#{id}: Num messages: {len(state.values["messages"])}, Next node: {state.next}")
                    print("-" * 80)
                id_to_replay = int(input("ID of the state to replay: "))
                state_to_replay = list(graph.get_state_history(config))[id_to_replay]
                state_config = state_to_replay.config
                stream_graph_updates(None, config=state_config, stream_mode="values")
            else:
                stream_graph_updates(user_input, config=config, stream_mode="values")
    except IOError:
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input, config=config)
        break
