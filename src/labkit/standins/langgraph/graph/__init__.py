"""StateGraph: nodes that read and update a shared state, joined by edges."""

from collections import namedtuple
from typing import get_type_hints

from langgraph._practice import NotInPracticeVersion, missing, module_getattr
from langgraph.errors import _HELP, GraphRecursionError, InvalidUpdateError

START, END = "__start__", "__end__"
DEFAULT_RECURSION_LIMIT = 10007

Node = namedtuple("Node", "id name data")
Edge = namedtuple("Edge", "source target data conditional")
Branch = namedtuple("Branch", "path ends")


def _coerce_error(thing):
    return TypeError(f"Expected a Runnable, callable or dict.Instead got an unsupported type: {type(thing)}")


def _wants_config(function):
    try:
        names = list(__import__("inspect").signature(function).parameters)
    except (TypeError, ValueError):
        return False
    return len(names) > 1 and names[1] == "config"


class StateGraph:
    """Build a graph: add_node, add_edge, add_conditional_edges, then compile()."""

    def __init__(self, state_schema, **kwargs):
        if kwargs:
            raise TypeError(missing(f"the {next(iter(kwargs))} argument of StateGraph"))
        try:
            hints = get_type_hints(state_schema, include_extras=True)
        except Exception:
            hints = {}
        if not (isinstance(state_schema, type) and issubclass(state_schema, dict) and hints):
            raise TypeError(missing("a state that is not a TypedDict class with fields") + " Define the state as `class MyState(TypedDict):` with one line for each field.")
        self.state_schema = state_schema
        self.channels = {}                                    # field name -> the function that merges a new value in, or None
        for key, hint in hints.items():
            extras = [extra for extra in getattr(hint, "__metadata__", ()) if callable(extra)]
            self.channels[key] = (extras[-1], getattr(hint, "__origin__", None)) if extras else None
        self.nodes, self.edges, self.branches = {}, set(), {}
        self._edge_order = []

    def add_node(self, node, action=None, **kwargs):
        if kwargs:
            raise TypeError(missing(f"the {next(iter(kwargs))} argument of add_node"))
        if action is None:
            node, action = getattr(node, "__name__", None), node
        if not isinstance(node, str):
            raise TypeError(f"A node needs a name: add_node(\"name\", function). Got {node!r}.")
        if node in self.nodes:
            raise ValueError(f"Node `{node}` already present.")
        if node in (START, END):
            raise ValueError(f"Node `{node}` is reserved.")
        if not callable(action) and not hasattr(action, "invoke"):
            raise _coerce_error(action)
        self.nodes[node] = action
        return self

    def add_edge(self, start_key, end_key):
        if not isinstance(start_key, str):
            raise TypeError(missing("an edge that starts from a list of nodes"))
        if start_key == END:
            raise ValueError("END cannot be a start node")
        if end_key == START:
            raise ValueError("START cannot be an end node")
        if (start_key, end_key) not in self.edges:
            self.edges.add((start_key, end_key))
            self._edge_order.append((start_key, end_key))
        return self

    def add_conditional_edges(self, source, path, path_map=None):
        name = getattr(path, "__name__", None)
        name = None if name == "<lambda>" else name
        if not callable(path):
            raise _coerce_error(path)
        if (name or "condition") in self.branches.get(source, {}):
            raise ValueError(f"Branch with name `{name}` already exists for node `{source}`")
        if path_map is not None and not isinstance(path_map, dict):
            path_map = {target: target for target in path_map}
        self.branches.setdefault(source, {})[name or "condition"] = Branch(path, path_map)
        return self

    def set_entry_point(self, key):
        return self.add_edge(START, key)

    def set_finish_point(self, key):
        return self.add_edge(key, END)

    def set_conditional_entry_point(self, path, path_map=None):
        return self.add_conditional_edges(START, path, path_map)

    def __getattr__(self, name):
        if name in ("add_sequence", "validate"):
            raise NotInPracticeVersion(missing(f"StateGraph.{name}"))
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def compile(self, checkpointer=None, **kwargs):
        if kwargs:
            raise TypeError(missing(f"the {next(iter(kwargs))} argument of compile"))
        sources = {start for start, _ in self.edges} | set(self.branches)
        for source in sources:
            if source != START and source not in self.nodes:
                raise ValueError(f"Found edge starting at unknown node '{source}'")
        if START not in sources:
            raise ValueError("Graph must have an entrypoint: add at least one edge from START to another node")
        for _, target in self._edge_order:
            if target != END and target not in self.nodes:
                raise ValueError(f"Found edge ending at unknown node `{target}`")
        for source, branches in self.branches.items():
            for name, branch in branches.items():
                for target in (branch.ends or {}).values():
                    if target != END and target not in self.nodes:
                        raise ValueError(f"At '{source}' node, '{name}' branch found unknown target '{target}'")
        return CompiledStateGraph(self, checkpointer)


class DrawnGraph:
    """What get_graph() returns: the nodes and edges, for looking at the shape of a graph."""

    def __init__(self, builder):
        self.nodes = {name: Node(name, name, builder.nodes.get(name)) for name in [START, *builder.nodes, END]}
        self.edges = []
        waiting, reached = [START], {START}
        while waiting:                       # like the real LangGraph, draw only what can be reached from the start
            source = waiting.pop(0)
            found = [Edge(start, end, None, False) for start, end in builder._edge_order if start == source]
            for branch in builder.branches.get(source, {}).values():
                found += [Edge(source, end, None if key == end else key, True) for key, end in (branch.ends or {}).items()]
            self.edges += found or [Edge(source, END, None, False)]
            for edge in found:
                if edge.target != END and edge.target not in reached:
                    reached.add(edge.target)
                    waiting.append(edge.target)

    def __getattr__(self, name):
        if name.startswith("draw_") or name in ("print_ascii", "to_json"):
            raise NotInPracticeVersion(missing(f"get_graph().{name}()"))
        raise AttributeError(f"'Graph' object has no attribute '{name}'")


class CompiledStateGraph:
    """A graph that is ready to run with invoke() or stream()."""

    def __init__(self, builder, checkpointer=None):
        self.builder, self.checkpointer = builder, checkpointer
        self.nodes = dict(builder.nodes)

    def __getattr__(self, name):
        if name in ("ainvoke", "astream", "get_state", "get_state_history", "update_state", "batch", "with_config", "astream_events"):
            raise NotInPracticeVersion(missing(f".{name}() on a compiled graph"))
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def get_graph(self, *args, **kwargs):
        return DrawnGraph(self.builder)

    def _merge(self, state, key, value):
        merger = self.builder.channels[key]
        if merger is None:
            state[key] = value
            return
        function, kind = merger
        if key not in state:
            try:
                state[key] = kind()
            except Exception:
                state[key] = value
                return
        state[key] = function(state[key], value)

    def _writes(self, update):
        if update is None:
            return {}
        if not isinstance(update, dict):
            raise InvalidUpdateError(f"Expected dict, got {update}{_HELP}INVALID_GRAPH_NODE_RETURN_VALUE")
        return {key: value for key, value in update.items() if key in self.builder.channels}

    def _targets(self, source, state):
        found = [end for start, end in self.builder._edge_order if start == source]
        for branch in self.builder.branches.get(source, {}).values():
            chosen = branch.path(dict(state))
            for key in chosen if isinstance(chosen, (list, tuple)) else [chosen]:
                found.append(branch.ends[key] if branch.ends is not None else key)   # a KeyError here means the route name is not in the mapping
        return [node for node in found if node in self.nodes]

    def _output(self, state):
        return {key: state[key] for key in self.builder.channels if key in state}

    def _run(self, input, config):
        """Run the graph one step at a time. Yields ("updates", {node: update}) and ("values", state)."""
        config = dict(config or {})
        limit = config.get("recursion_limit", DEFAULT_RECURSION_LIMIT)
        state, thread = {}, None
        if self.checkpointer is not None:
            thread = (config.get("configurable") or {}).get("thread_id")
            if thread is None:
                raise ValueError("Checkpointer requires one or more of the following 'configurable' keys: thread_id, checkpoint_ns, checkpoint_id")
            state = dict(self.checkpointer.storage.get(thread, {}))
        for key, value in self._writes(input).items():
            self._merge(state, key, value)
        yield "values", self._output(state)
        active, steps = self._targets(START, state), 0
        while True:
            if steps >= limit:
                raise GraphRecursionError(limit)
            if not active:
                break
            steps += 1
            config["__practice__"] = {"step": steps, "limit": limit}
            before, done = dict(state), []
            for name in [node for node in self.nodes if node in active]:
                action = self.nodes[name]
                call = action.invoke if hasattr(action, "invoke") else action
                update = call(dict(before), config) if _wants_config(call) else call(dict(before))
                done.append((name, update, self._writes(update)))
            written = set()
            for name, update, writes in done:
                for key, value in writes.items():
                    if self.builder.channels[key] is None and key in written:
                        raise InvalidUpdateError(f"At key '{key}': Can receive only one value per step. Use an Annotated key to handle multiple values.{_HELP}INVALID_CONCURRENT_GRAPH_UPDATE")
                    written.add(key)
                    self._merge(state, key, value)
            active = []
            for name, update, writes in done:
                seen = dict(before)                       # a routing function sees the state with its own node's update
                for key, value in writes.items():
                    self._merge(seen, key, value)
                active += self._targets(name, seen)
                yield "updates", {name: update}
            yield "values", self._output(state)
        if thread is not None:
            self.checkpointer.storage[thread] = dict(state)

    def stream(self, input, config=None, *, stream_mode="updates", **kwargs):
        if kwargs:
            raise TypeError(missing(f"the {next(iter(kwargs))} argument of stream"))
        if stream_mode not in ("updates", "values"):
            raise ValueError(missing(f"stream_mode={stream_mode!r}"))
        for kind, payload in self._run(input, config):
            if kind == stream_mode:
                yield payload

    def invoke(self, input, config=None, *, stream_mode="values", **kwargs):
        chunks = list(self.stream(input, config, stream_mode=stream_mode, **kwargs))
        return chunks[-1] if stream_mode == "values" else chunks


__getattr__ = module_getattr("langgraph.graph", ("message",), {"add_messages": "langgraph.graph.message", "MessagesState": "langgraph.graph.message"})
