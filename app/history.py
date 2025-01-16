from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
import pprint
import json
import uuid

in_memory_store = InMemoryStore()

user_id = "1"
namespace_for_memory = (user_id, "memories")

memory_id = str(uuid.uuid4())
memory = {"food_preference" : "私はピザが好きです。"}
in_memory_store.put(namespace_for_memory, memory_id, memory)

memories = in_memory_store.search(namespace_for_memory)
print(json.dumps(memories[-1].dict(), indent=4))

# class State(TypedDict):
#     foo: int
#     bar: Annotated[list[str], add]

# def node_a(state: State):
#     return {"foo": "１回目です", "bar": ["１回目です"]}

# def node_b(state: State):
#     return {"foo": "２回目です", "bar": ["２回目です"]}

# workflow = StateGraph(State)
# workflow.add_node(node_a)
# workflow.add_node(node_b)
# workflow.add_edge(START, "node_a")
# workflow.add_edge("node_a", "node_b")
# workflow.add_edge("node_b", END)

# checkpointer = MemorySaver()
# graph = workflow.compile(checkpointer=checkpointer)

# config = {"configurable": {"thread_id": "1"}}
# graph.get_state(config)

# graph.invoke({"foo": ""}, config)

# 保持している最後のstateをjson形式で出力
# print(json.dumps(graph.get_state(config), indent=4))

# stateの履歴をjson形式で出力(history.txt)
# print(json.dumps(list(graph.get_state_history(config)), indent=4, ensure_ascii=False))
