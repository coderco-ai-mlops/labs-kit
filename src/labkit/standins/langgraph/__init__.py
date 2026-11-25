"""A practice version of LangGraph for the labs site, where the real package cannot be installed.

It covers what labs s5-10 and s5-11 use: StateGraph with nodes, edges and conditional edges,
compile, invoke and stream, create_react_agent with tools, and memory with InMemorySaver.
"""

from langgraph._practice import module_getattr, watch_imports

__getattr__ = module_getattr("langgraph", ("graph", "prebuilt", "checkpoint", "errors"))
watch_imports("langgraph")
