"""The errors the practice version of LangGraph raises. They have the same names as in the real LangGraph."""

_HELP = "\nFor troubleshooting, visit: https://docs.langchain.com/oss/python/langgraph/errors/"


class GraphRecursionError(RecursionError):
    """The graph ran for more steps than its recursion limit allows."""

    def __init__(self, limit):
        super().__init__(f"Recursion limit of {limit} reached without hitting a stop condition. "
                         f"You can increase the limit by setting the `recursion_limit` config key.{_HELP}GRAPH_RECURSION_LIMIT")


class InvalidUpdateError(Exception):
    """A node returned something the graph cannot merge into the state."""
