"""Fake chat models: they look like a real chat model to a chain, and reply with text you give them."""

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from langchain_core._practice import missing, module_getattr
from langchain_core.messages import AIMessage, BaseMessage, convert_to_messages
from langchain_core.prompts import ChatPromptValue
from langchain_core.runnables import Runnable, left_out


class BaseChatModel(BaseModel, Runnable):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            raise left_out(self, name) from None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError

    def invoke(self, input, config=None, *, stop=None, **kwargs):
        if isinstance(input, ChatPromptValue):
            messages = input.to_messages()
        elif isinstance(input, str):
            messages = convert_to_messages([input])
        elif isinstance(input, (list, tuple)):
            messages = convert_to_messages(input)
        else:
            raise ValueError(f"Invalid input type {type(input)}. Must be a PromptValue, str, or list of BaseMessages.")
        reply = self._generate(messages, stop=stop, run_manager=None, **kwargs)
        if reply.id is None:
            reply.id = f"lc_run--{uuid.uuid4()}-0"
        return reply

    def bind_tools(self, tools, **kwargs):
        raise NotImplementedError(
            "This fake model does not accept tools. Make a subclass with a method `bind_tools(self, tools, **kwargs)` that returns `self`."
        )


class FakeListChatModel(BaseChatModel):
    """Replies with each text in `responses` in turn, then starts again from the first."""

    responses: list[str]
    i: int = Field(default=0, repr=False)

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        reply = self.responses[self.i]
        self.i = self.i + 1 if self.i < len(self.responses) - 1 else 0
        return reply

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return AIMessage(content=self._call(messages, stop=stop, run_manager=run_manager, **kwargs))


class FakeMessagesListChatModel(BaseChatModel):
    """Replies with each message in `responses` in turn, then starts again from the first."""

    responses: list[Any]
    i: int = Field(default=0, repr=False)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        reply = self.responses[self.i]
        self.i = self.i + 1 if self.i < len(self.responses) - 1 else 0
        if not isinstance(reply, BaseMessage):
            raise TypeError(f"FakeMessagesListChatModel replies with messages, such as AIMessage(content=\"...\"). It was given {reply!r}.")
        return reply


class GenericFakeChatModel(BaseChatModel):
    """Replies with the next item of `messages`, which is an iterator: GenericFakeChatModel(messages=iter([...]))."""

    messages: Any

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        reply = next(self.messages)
        return AIMessage(content=reply) if isinstance(reply, str) else reply


__getattr__ = module_getattr("langchain_core.language_models.fake_chat_models")
