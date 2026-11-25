"""Output parsers: the last step of a chain, which turns the model's message into something simpler."""

from langchain_core._practice import module_getattr
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable


class StrOutputParser(Runnable):
    """Takes the model's message and returns its text."""

    def invoke(self, input, config=None, **kwargs):
        if isinstance(input, BaseMessage):
            return input.text
        if isinstance(input, str):
            return input
        raise TypeError(f"StrOutputParser expects the model's message or a piece of text. It was given a {type(input).__name__}: {input!r}")

    def __repr__(self):
        return "StrOutputParser()"


__getattr__ = module_getattr("langchain_core.output_parsers")
