"""A practice version of langchain-core for the labs site, where the real package cannot be installed.

It covers what labs s5-09 and s5-11 use: chat prompt templates, chains joined with `|`,
StrOutputParser, messages, the fake chat models and the @tool decorator.
"""

from langchain_core._practice import module_getattr, watch_imports

__version__ = "1.8.4"

__getattr__ = module_getattr("langchain_core", ("messages", "prompts", "runnables", "output_parsers", "tools", "language_models"))
watch_imports("langchain_core")
