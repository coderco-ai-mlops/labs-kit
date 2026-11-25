"""Prompt templates: text with named blanks, filled in when the chain runs."""

from string import Formatter

from langchain_core._practice import missing, module_getattr
from langchain_core.messages import BaseMessage, message_class
from langchain_core.runnables import Runnable

_HELP = "\nFor troubleshooting, visit: https://docs.langchain.com/oss/python/langchain/errors/INVALID_PROMPT_INPUT "


def _blanks(template):
    return [name for _, name, _, _ in Formatter().parse(template) if name is not None]


class ChatPromptValue:
    """A filled-in prompt: the messages that will be sent to the model."""

    def __init__(self, messages):
        self.messages = list(messages)

    def to_messages(self):
        return list(self.messages)

    def to_string(self):
        return "\n".join(f"{ {'human': 'Human', 'ai': 'AI', 'system': 'System'}.get(m.type, m.type)}: {m.content}" for m in self.messages)

    def __repr__(self):
        return f"ChatPromptValue(messages={self.messages!r})"

    def __eq__(self, other):
        return isinstance(other, ChatPromptValue) and other.messages == self.messages


class ChatPromptTemplate(Runnable):
    def __init__(self, messages, **kwargs):
        if kwargs:
            raise TypeError(missing(f"the {next(iter(kwargs))} argument of ChatPromptTemplate"))
        self.messages = []
        for item in messages:
            if isinstance(item, str):
                item = ("human", item)
            if isinstance(item, BaseMessage):
                self.messages.append(item)
            elif isinstance(item, (tuple, list)) and len(item) == 2 and all(isinstance(part, str) for part in item):
                if item[0] == "placeholder":
                    raise ValueError(missing("placeholder messages in a prompt"))
                message_class(item[0])
                self.messages.append((item[0], item[1]))
            else:
                raise ValueError(f"A prompt message is a (role, text) pair such as (\"human\", \"Explain {{topic}}\"), a string or a message. Got {item!r}.")
        names = {name for item in self.messages if isinstance(item, tuple) for name in _blanks(item[1])}
        self.input_variables = sorted(names)

    @classmethod
    def from_template(cls, template, **kwargs):
        if not isinstance(template, str):
            raise TypeError(f"from_template takes the prompt as text. Got a {type(template).__name__}.")
        return cls([("human", template)], **kwargs)

    @classmethod
    def from_messages(cls, messages, **kwargs):
        return cls(messages, **kwargs)

    def format_messages(self, **values):
        return [item if isinstance(item, BaseMessage) else message_class(item[0])(item[1].format(**values)) for item in self.messages]

    def format(self, **values):
        return ChatPromptValue(self.format_messages(**values)).to_string()

    def invoke(self, input, config=None, **kwargs):
        if not isinstance(input, dict):
            if len(self.input_variables) != 1:
                raise TypeError(f"Expected mapping type as input to ChatPromptTemplate. Received {type(input)}.{_HELP}")
            input = {self.input_variables[0]: input}
        absent = set(self.input_variables) - set(input)
        if absent:
            example = sorted(absent)[0]
            raise KeyError(
                f"Input to ChatPromptTemplate is missing variables {absent}.  Expected: {self.input_variables} Received: {list(input)}"
                f"\nNote: if you intended {{{example}}} to be part of the string and not a variable, "
                f"please escape it with double curly braces like: '{{{{{example}}}}}'.{_HELP}"
            )
        return ChatPromptValue(self.format_messages(**input))

    def __repr__(self):
        return f"ChatPromptTemplate(input_variables={self.input_variables!r}, messages={self.messages!r})"


__getattr__ = module_getattr("langchain_core.prompts")
