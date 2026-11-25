"""The messages a chat model reads and writes."""

from langchain_core._practice import missing, module_getattr


class BaseMessage:
    type = "base"
    _shown = ("additional_kwargs", "response_metadata", "name", "id")

    def __init__(self, content=None, **fields):
        if not isinstance(content, (str, list)):
            raise TypeError(f"A message needs its text: {type(self).__name__}(content=\"...\"). It was given {content!r}.")
        self.content = content
        self.additional_kwargs = fields.pop("additional_kwargs", {})
        self.response_metadata = fields.pop("response_metadata", {})
        self.name = fields.pop("name", None)
        self.id = fields.pop("id", None)
        fields.pop("type", None)
        self.__dict__.update(fields)

    @property
    def text(self):
        return self.content if isinstance(self.content, str) else "".join(str(part) for part in self.content)

    def _fields(self):
        names = ("content",) + tuple(n for n in self._shown if n not in ("name", "id") or getattr(self, n) is not None)
        return [(name, getattr(self, name)) for name in names]

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in self._fields())})"

    def __str__(self):
        return " ".join(f"{k}={v!r}" for k, v in self._fields())

    def __eq__(self, other):
        return type(other) is type(self) and vars(other) == vars(self)

    def pretty_repr(self, html=False):
        title = f" {self.type.title()} Message "
        left = (80 - len(title)) // 2
        lines = ["=" * left + title + "=" * (80 - len(title) - left)]
        if self.name:
            lines.append(f"Name: {self.name}")
        for call in getattr(self, "tool_calls", []):
            lines.append(f"Tool Calls:\n  {call['name']} ({call.get('id')})\n  Args: {call['args']}")
        return "\n".join(lines) + f"\n\n{self.content}"

    def pretty_print(self):
        print(self.pretty_repr())


class HumanMessage(BaseMessage):
    type = "human"


class SystemMessage(BaseMessage):
    type = "system"


class AIMessage(BaseMessage):
    """A reply from the model. `tool_calls` lists the tools the model is asking to be run."""

    type = "ai"
    _shown = BaseMessage._shown + ("tool_calls", "invalid_tool_calls")

    def __init__(self, content=None, **fields):
        calls = fields.pop("tool_calls", None) or []
        self.invalid_tool_calls = fields.pop("invalid_tool_calls", [])
        super().__init__(content, **fields)
        self.tool_calls = []
        for call in calls:
            if not isinstance(call, dict) or "name" not in call or "args" not in call:
                raise ValueError(f'Each tool call is a dictionary like {{"name": "add", "args": {{"a": 1}}, "id": "call-1"}}. Got {call!r}.')
            self.tool_calls.append({"name": call["name"], "args": call["args"], "id": call.get("id"), "type": "tool_call"})


class ToolMessage(BaseMessage):
    """The result of running one tool, sent back to the model."""

    type = "tool"
    _shown = ("name", "id", "tool_call_id")

    def __init__(self, content=None, **fields):
        if "tool_call_id" not in fields:
            raise KeyError("tool_call_id")
        self.tool_call_id = fields.pop("tool_call_id")
        self.status = fields.pop("status", "success")
        super().__init__(content if isinstance(content, (str, list)) else str(content), **fields)


_ROLES = {"human": HumanMessage, "user": HumanMessage, "ai": AIMessage, "assistant": AIMessage, "system": SystemMessage, "tool": ToolMessage}


def message_class(role):
    if role not in _ROLES or role == "tool":
        raise ValueError(f"Unexpected message type: {role}. Use one of 'human', 'user', 'ai', 'assistant', or 'system'.")
    return _ROLES[role]


def convert_to_messages(items):
    """Turn text, (role, text) pairs and {"role": ..., "content": ...} dictionaries into message objects."""
    found = []
    for item in items:
        if isinstance(item, BaseMessage):
            found.append(item)
        elif isinstance(item, str):
            found.append(HumanMessage(item))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            found.append(message_class(item[0])(item[1]))
        elif isinstance(item, dict) and ("role" in item or "type" in item) and "content" in item:
            fields = {k: v for k, v in item.items() if k not in ("role", "type", "content")}
            role = item.get("role", item.get("type"))
            found.append((ToolMessage if role == "tool" else message_class(role))(item["content"], **fields))
        else:
            raise NotImplementedError(f"Unsupported message type: {type(item)}. Use a message object, a (role, text) pair or a dictionary with \"role\" and \"content\".")
    return found


__getattr__ = module_getattr("langchain_core.messages")
