"""Shared by the practice version of LangGraph: the message shown for anything it leaves out."""


def missing(what):
    return (f"The practice version of LangGraph on the labs site does not include {what}. "
            "It is in the real LangGraph: see Go further in this lab.")


class NotInPracticeVersion(AttributeError):
    """Raised for a method or attribute the practice version leaves out. The message says so in a plain sentence."""


class _MissingModules:
    """Asked last, when Python cannot find a module such as `langgraph.something`: says so in a plain sentence."""

    def __init__(self, package):
        self.package = package

    def find_spec(self, name, path=None, target=None):
        if name.startswith(self.package + "."):
            raise ImportError(missing(name))
        return None


def watch_imports(package):
    import sys
    if not any(getattr(finder, "package", None) == package for finder in sys.meta_path):
        sys.meta_path.append(_MissingModules(package))


def module_getattr(module, submodules=(), lazy=None):
    """What a module does when asked for a name it does not have: say so in a plain sentence."""
    def __getattr__(name):
        if name.startswith("__"):
            raise AttributeError(f"module '{module}' has no attribute '{name}'")
        import importlib
        if name in submodules:
            return importlib.import_module(f"{module}.{name}")
        if lazy and name in lazy:
            return getattr(importlib.import_module(lazy[name]), name)
        raise ImportError(missing(f"{module}.{name}"))   # ImportError, so the sentence also survives `from x import y`
    return __getattr__
