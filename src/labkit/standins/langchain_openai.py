"""A placeholder for langchain-openai on the labs site.

ChatOpenAI connects to a real model over the network, which is what live.py does on your
own machine. The labs on the site pass a fake chat model to your functions instead.
"""

_WHY = ("ChatOpenAI connects to a real model, and the labs site runs your code with a fake model instead. "
        "Use FakeListChatModel from langchain_core.language_models here. "
        "To use a real model, run live.py on your own machine: see Go further in this lab.")


class ChatOpenAI:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(_WHY)


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    raise ImportError(f"The practice version of LangChain on the labs site does not include langchain_openai.{name}. "
                         "It is in the real LangChain: see Go further in this lab.")
