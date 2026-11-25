"""Tools: functions a model can ask an agent to run."""

import inspect
import json
from typing import Any

from pydantic import create_model

from langchain_core._practice import missing, module_getattr
from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable


class BaseTool(Runnable):
    name: str = ""
    description: str = ""
    return_direct = False

    @property
    def args(self):
        return self.args_schema.model_json_schema()["properties"]

    @property
    def tool_call_schema(self):
        return self.args_schema

    def run(self, tool_input, **kwargs):
        return self.invoke(tool_input)

    def invoke(self, input, config=None, **kwargs):
        call = input if isinstance(input, dict) and input.get("type") == "tool_call" else None
        values = call["args"] if call else input
        if isinstance(values, str):
            first = next(iter(self.args_schema.model_fields), None)
            values = {first: values} if first else {}
        if not isinstance(values, dict):
            raise TypeError(f"Run a tool with a dictionary of its arguments, such as {self.name}.invoke({{...}}). It was given {values!r}.")
        checked = self.args_schema.model_validate(values)
        result = self.func(**{key: getattr(checked, key) for key in self.args_schema.model_fields if key in values})
        if not call:
            return result
        return ToolMessage(content=as_text(result), name=self.name, tool_call_id=call.get("id"))


class StructuredTool(BaseTool):
    def __init__(self, func, name=None, description=None):
        description = description or (func.__doc__ or "").strip()
        if not description:
            raise ValueError("Function must have a docstring if description not provided.")
        self.func, self.name, self.description = func, name or func.__name__, description
        fields = {}
        for arg in inspect.signature(func).parameters.values():
            if arg.kind in (arg.VAR_POSITIONAL, arg.VAR_KEYWORD):
                raise TypeError(missing("tools that take *args or **kwargs"))
            kind = Any if arg.annotation is arg.empty else arg.annotation
            fields[arg.name] = (kind, ... if arg.default is arg.empty else arg.default)
        self.args_schema = create_model(self.name, __doc__=description, **fields)

    def __repr__(self):
        return f"StructuredTool(name={self.name!r}, description={self.description!r}, args_schema={self.args_schema!r}, func={self.func!r})"


def as_text(result):
    """How a tool's result is written into a ToolMessage."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(result)


def tool(name_or_callable=None, *, description=None, **kwargs):
    """Turn a function into a tool. Put @tool on the line above `def`."""
    if kwargs:
        raise TypeError(missing(f"the {next(iter(kwargs))} argument of @tool"))
    if callable(name_or_callable):
        return StructuredTool(name_or_callable, description=description)
    if name_or_callable is not None and not isinstance(name_or_callable, str):
        raise TypeError(f"@tool goes above a function, or takes the tool's name as text. Got {name_or_callable!r}.")
    return lambda func: StructuredTool(func, name=name_or_callable, description=description)


__getattr__ = module_getattr("langchain_core.tools")
