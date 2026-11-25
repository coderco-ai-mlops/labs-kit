"""Memory for a graph: the state of each conversation, kept under its thread_id while the program runs."""

from langgraph._practice import module_getattr


class InMemorySaver:
    def __init__(self):
        self.storage = {}   # thread_id -> the state at the end of that thread's last run


MemorySaver = InMemorySaver

__getattr__ = module_getattr("langgraph.checkpoint.memory")
