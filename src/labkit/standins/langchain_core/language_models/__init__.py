"""Chat models. On the labs site these are the fake models, which give replies you choose."""

from langchain_core._practice import module_getattr
from langchain_core.language_models.fake_chat_models import (
    BaseChatModel,
    FakeListChatModel,
    FakeMessagesListChatModel,
    GenericFakeChatModel,
)

__getattr__ = module_getattr("langchain_core.language_models", ("fake_chat_models",))
