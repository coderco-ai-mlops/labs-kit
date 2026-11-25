"""create_react_agent: a ready-made graph in which a model decides which tools to call."""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, as_text, tool as make_tool
from pydantic import ValidationError

from langgraph._practice import missing, module_getattr
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


class ToolNode:
    """Runs every tool the last AI message asks for, and returns one ToolMessage for each."""

    def __init__(self, tools, **kwargs):
        if kwargs:
            raise TypeError(missing(f"the {next(iter(kwargs))} argument of ToolNode"))
        tools = [each if isinstance(each, BaseTool) else make_tool(each) for each in tools]
        self.tools_by_name = {each.name: each for each in tools}

    def _run_one(self, call):
        name, args = call["name"], call["args"]
        fields = {"name": name, "tool_call_id": call.get("id")}
        if name not in self.tools_by_name:
            return ToolMessage(content=f"Error: {name} is not a valid tool, try one of [{', '.join(self.tools_by_name)}].", status="error", **fields)
        each = self.tools_by_name[name]
        try:
            each.args_schema.model_validate(args)
        except ValidationError as err:      # wrong arguments are reported to the model, so it can try again
            lines = "\n".join(f" {'.'.join(str(part) for part in e['loc'])}: {e['msg']}" for e in err.errors())
            return ToolMessage(content=f"Error invoking tool '{name}' with kwargs {args} with error:\n{lines}\n Please fix the error and try again.", status="error", **fields)
        return ToolMessage(content=as_text(each.invoke(args)), **fields)

    def __call__(self, state):
        last = state["messages"][-1]
        return {"messages": [self._run_one(call) for call in getattr(last, "tool_calls", [])]}

    invoke = __call__


def tools_condition(state):
    """Go to the tools node when the last AI message asks for a tool, and finish otherwise."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def create_react_agent(model, tools, *, prompt=None, checkpointer=None, **kwargs):
    """Build an agent: the model is called, the tools it asks for are run, and the results go back to it, until it answers."""
    if kwargs:
        raise TypeError(missing(f"the {next(iter(kwargs))} argument of create_react_agent"))
    if isinstance(model, str):
        raise TypeError(missing("choosing a model by name") + " Pass a chat model object.")
    if prompt is not None and not isinstance(prompt, (str, SystemMessage)):
        raise TypeError(missing("a prompt that is not text or a SystemMessage"))
    tool_node = ToolNode(tools)
    bound = model.bind_tools(list(tool_node.tools_by_name.values())) if tool_node.tools_by_name else model

    def agent(state, config):
        messages = list(state.get("messages", []))
        if prompt is not None:
            messages = [SystemMessage(prompt) if isinstance(prompt, str) else prompt] + messages
        reply = bound.invoke(messages)
        run = config["__practice__"]
        if reply.tool_calls and run["limit"] - run["step"] < 2:
            reply = AIMessage(content="Sorry, need more steps to process this request.", id=reply.id)
        return {"messages": [reply]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


__getattr__ = module_getattr("langgraph.prebuilt")
