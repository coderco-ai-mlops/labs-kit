from langgraph._practice import module_getattr

__getattr__ = module_getattr("langgraph.checkpoint", ("memory",))
