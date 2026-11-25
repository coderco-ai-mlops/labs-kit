"""Helpers for writing a lab's checks.py.

A check is a function that raises CheckFail (or fails an assert) with a message
written for the member. The title says what is being checked.

    from labkit import check

    @check("city holds the name of your city")
    def _(lab):
        s = lab.script()
        s.need("city", str)
"""

import ast
import builtins
import contextlib
import io
import re
import traceback
from pathlib import Path

_REGISTRY: list = []


class CheckFail(AssertionError):
    """A check did not pass. The message is shown to the member as written."""


def check(title: str, hint: str = ""):
    def register(fn):
        _REGISTRY.append((title, hint, fn))
        return fn

    return register


def registered() -> list:
    return list(_REGISTRY)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class ScriptResult:
    """What happened when the member's file ran: its variables, what it printed, and any error."""

    def __init__(self, name: str, variables: dict, out: str, error: str | None):
        self.name = name
        self.vars = variables
        self.out = out
        self.lines = [line.rstrip() for line in out.splitlines()]
        self.error = error

    def ok(self) -> None:
        if self.error:
            raise CheckFail(self.error)

    def need(self, var: str, kind=None):
        """Return a variable (or function) from the file, or fail with a plain message."""
        self.ok()
        if var not in self.vars:
            raise CheckFail(f"Create `{var}` in {self.name}. It was not found.")
        value = self.vars[var]
        if kind is not None:
            kinds = kind if isinstance(kind, tuple) else (kind,)
            wrong_bool = isinstance(value, bool) and bool not in kinds
            if wrong_bool or not isinstance(value, kinds):
                want = " or ".join(k.__name__ for k in kinds)
                raise CheckFail(f"`{var}` should be a {want}. It is a {type(value).__name__}: {value!r}")
        return value

    def need_number(self, var: str):
        value = self.need(var)
        if not _is_number(value):
            raise CheckFail(f"`{var}` should be a number. It is a {type(value).__name__}: {value!r}")
        return value

    def need_function(self, var: str):
        value = self.need(var)
        if not callable(value):
            raise CheckFail(f"`{var}` should be a function. Define it with `def {var}(...):`")
        return value

    def printed(self, text) -> bool:
        """True when the text appears anywhere in what the file printed."""
        return str(text) in self.out

    def need_printed(self, text, what: str = "") -> None:
        self.ok()
        if not self.printed(text):
            label = what or f"`{text}`"
            shown = self.out.strip() or "(nothing)"
            raise CheckFail(f"Expected {label} in the output. Your file printed:\n{_indent(shown)}")

    def need_line(self, text, what: str = "") -> None:
        """The text must be a whole printed line, not part of one."""
        self.ok()
        if str(text) not in self.lines:
            label = what or f"`{text}`"
            shown = self.out.strip() or "(nothing)"
            raise CheckFail(f"Expected a printed line that is exactly {label}. Your file printed:\n{_indent(shown)}")


def _indent(text: str) -> str:
    lines = text.splitlines()
    if len(lines) > 12:
        lines = lines[:12] + [f"... ({len(lines) - 12} more lines)"]
    return "\n".join("    " + line for line in lines)


class _SetValues(ast.NodeTransformer):
    """Replaces the first top-level assignment to each named variable with a fixed value."""

    def __init__(self, values: dict):
        self.values = dict(values)
        self.done: set = set()

    def visit_Module(self, node):
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                name = stmt.targets[0].id
                if name in self.values and name not in self.done:
                    stmt.value = ast.copy_location(ast.Constant(self.values[name]), stmt.value)
                    self.done.add(name)
        return node


def _no_input(*_args, **_kwargs):
    raise RuntimeError("input() is not used in this lab. Put the value in a variable instead.")


class Lab:
    """Handed to every check. Runs the member's files and remembers the results."""

    def __init__(self, folder: Path, default_file: str = "exercise.py"):
        self.folder = Path(folder)
        self.default_file = default_file
        self._cache: dict = {}
        self.faults: set = set()  # messages that mean a file would not run at all

    def path(self, name: str | None = None) -> Path:
        return self.folder / (name or self.default_file)

    def source(self, name: str | None = None) -> str:
        return self.path(name).read_text(encoding="utf-8")

    def script(self, name: str | None = None, set: dict | None = None) -> ScriptResult:
        """Run a file and return what it did.

        `set` reruns the file with different starting values, for example
        lab.script(set={"age": 17}), so a check can test more than the value the member typed.
        """
        name = name or self.default_file
        key = (name, tuple(sorted(set.items())) if set else None)
        if key not in self._cache:
            self._cache[key] = self._run(name, set)
            error = self._cache[key].error or ""
            if "stopped with an error" in error or "syntax error" in error:
                self.faults.add(error)
        return self._cache[key]

    def _run(self, name: str, values: dict | None) -> ScriptResult:
        path = self.path(name)
        if not path.exists():
            return ScriptResult(name, {}, "", f"{name} is missing. Run `lab reset` for this lab to get it back.")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=name)
        except SyntaxError as err:
            return ScriptResult(name, {}, "", _syntax(err, name))
        if values:
            setter = _SetValues(values)
            tree = ast.fix_missing_locations(setter.visit(tree))
            missing = [v for v in values if v not in setter.done]
            if missing:
                # If the file is broken anyway, say that: it is the thing to fix first.
                plain = self.script(name)
                problem = plain.error or f"Create a variable called `{missing[0]}` at the top level of {name}."
                return ScriptResult(name, {}, "", problem)
        namespace: dict = {"__name__": "__main__", "__file__": str(path)}
        out = io.StringIO()
        error = None
        real_input = builtins.input
        builtins.input = _no_input
        try:
            with contextlib.redirect_stdout(out):
                exec(compile(tree, name, "exec"), namespace)
        except BaseException as err:  # the member's code can raise anything, including SystemExit
            error = _describe(err, name)
        finally:
            builtins.input = real_input
        return ScriptResult(name, namespace, out.getvalue(), error)

    def call(self, fn, *args, **kwargs):
        """Call one of the member's functions, hiding anything it prints, with a plain message if it raises."""
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return fn(*args, **kwargs)
        except Exception as err:
            shown = ", ".join([repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()])
            raise CheckFail(f"`{fn.__name__}({shown})` raised {type(err).__name__}: {err}") from None


def _describe(err: BaseException, name: str) -> str:
    line = None
    for frame in traceback.extract_tb(err.__traceback__):
        if frame.filename == name:
            line = frame.lineno
    where = f" on line {line}" if line else ""
    return f"{name} stopped with an error{where}.\n{type(err).__name__}: {err}" + _plain(err)


def _plain(err: BaseException) -> str:
    """One or two plain sentences on what a common error means and what to try."""
    text, kind = str(err), type(err).__name__
    tip = ""
    if kind == "NameError":
        found = re.search(r"name '([^']+)'", text)
        word = found.group(1) if found else "that name"
        tip = (f'Python does not know the name {word}. If it is meant to be text, put it in quotes: "{word}". '
               "If it is a variable or function, check the spelling and that it is created above this line.")
    elif kind == "ZeroDivisionError":
        tip = "Something is being divided by zero. Check the value on the right of the division."
    elif kind == "IndexError":
        tip = "The code asked for a position that is not in the list. Positions start at 0, so a list of 3 items ends at position 2."
    elif kind == "KeyError":
        tip = f"The dictionary has no key {text}. Check the spelling and capital letters, or use .get() if the key may be missing."
    elif kind == "TypeError" and "can only concatenate str" in text:
        tip = "Text and a number are being joined with +. Use an f-string, or wrap the number in str()."
    elif kind == "TypeError" and "missing" in text and "argument" in text:
        tip = "A function was called without everything it needs. Compare the call with the function's first line."
    elif kind == "TypeError" and "unsupported operand" in text:
        tip = "Two values of different kinds are being combined. One of them may be text where a number is needed: look for stray quotes."
    elif kind == "AttributeError":
        tip = "That value does not have the thing after the dot. Check the spelling, and check the value is the kind you expect."
    elif kind == "ModuleNotFoundError":
        tip = "That module is not available here. Check the spelling of the import."
    elif kind == "RecursionError":
        tip = "A function keeps calling itself and never stops. It needs a case where it returns without calling itself."
    return f"\n\n{tip}" if tip else ""


_SYNTAX_TIPS = (
    ("unterminated string", "A piece of text is missing its closing quote."),
    ("was never closed", "A bracket was opened and never closed. Count the brackets on this line."),
    ("expected ':'", "This line needs a colon at the end. Lines starting with if, elif, else, for, while and def all end with one."),
    ("expected an indented block", "The line after a colon must be indented by four spaces."),
    ("unexpected indent", "This line is indented but should not be. Line it up with the lines around it."),
    ("unindent does not match", "The indentation here does not line up with the lines above. Use four spaces per level."),
    ("Maybe you meant '=='", "One equals sign stores a value. To compare two values, use two: ==."),
    ("Missing parentheses in call to 'print'", "print needs brackets: print(\"hello\")."),
    ("invalid syntax", "Python could not read this line. Look for a missing quote, bracket, comma or colon on it or on the line above."),
)


def _syntax(err: SyntaxError, name: str) -> str:
    message = f"{name} has a syntax error on line {err.lineno}.\n{type(err).__name__}: {err.msg}"
    for marker, tip in _SYNTAX_TIPS:
        if marker in (err.msg or ""):
            return f"{message}\n\n{tip}"
    return message
