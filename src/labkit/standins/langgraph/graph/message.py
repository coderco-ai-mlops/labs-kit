"""A state that holds a conversation: a list of messages that new messages are added to."""

import uuid
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, convert_to_messages

from langgraph._practice import module_getattr


def add_messages(left, right):
    """Add new messages to the list. A message whose id is already in the list replaces the old one."""
    merged = convert_to_messages(left if isinstance(left, (list, tuple)) else [left])
    for message in convert_to_messages(right if isinstance(right, (list, tuple)) else [right]):
        if message.id is None:
            message.id = str(uuid.uuid4())
    for message in merged:
        if message.id is None:
            message.id = str(uuid.uuid4())
    place = {message.id: index for index, message in enumerate(merged)}
    for message in convert_to_messages(right if isinstance(right, (list, tuple)) else [right]):
        if message.id in place:
            merged[place[message.id]] = message
        else:
            place[message.id] = len(merged)
            merged.append(message)
    return merged


class MessagesState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


__getattr__ = module_getattr("langgraph.graph.message")
