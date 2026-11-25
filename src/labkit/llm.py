"""Stand-ins for a language model and an embedding model, for checks.

Lab code that talks to a model takes the client as an argument. In a check we
pass a FakeLLM, so the check is free, needs no key and gives the same result
every time. In a real run the member passes a real OpenAI-compatible client
(Ollama, a free hosted tier, or a paid provider), which has the same shape.

    fake = FakeLLM(["Paris"])
    reply = ask(fake, "What is the capital of France?")
    assert fake.calls[0]["messages"][-1]["content"].endswith("France?")
"""

import hashlib
import math
import re


class _Obj:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def __repr__(self):
        return f"_Obj({self.__dict__})"


def _count(text) -> int:
    """A rough token count: good enough to practise cost maths on."""
    return max(1, len(str(text).split()))


class FakeLLM:
    """Looks like an OpenAI-compatible client: client.chat.completions.create(model=..., messages=[...]).

    `replies` are used in order. Each one is a string, an Exception to raise
    (to practise retries), or a function that takes the messages and returns a string.
    The last reply repeats if the code makes more calls than there are replies.
    Every call's keyword arguments are kept in `.calls`.
    """

    def __init__(self, replies=("ok",)):
        self.replies = list(replies) or ["ok"]
        self.calls: list = []
        self.chat = _Obj(completions=_Obj(create=self._create))

    def _create(self, **kwargs):
        if "model" not in kwargs or "messages" not in kwargs:
            raise TypeError("create() needs model=... and messages=[...]")
        messages = kwargs["messages"]
        if not isinstance(messages, list) or not all(isinstance(m, dict) and {"role", "content"} <= set(m) for m in messages):
            raise TypeError('messages must be a list of {"role": ..., "content": ...} dictionaries')
        self.calls.append(kwargs)
        reply = self.replies[min(len(self.calls), len(self.replies)) - 1]
        if isinstance(reply, BaseException):
            raise reply
        if callable(reply):
            reply = reply(messages)
        prompt_tokens = sum(_count(m["content"]) for m in messages)
        completion_tokens = _count(reply)
        return _Obj(
            model=kwargs["model"],
            choices=[_Obj(index=0, finish_reason="stop", message=_Obj(role="assistant", content=reply))],
            usage=_Obj(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens),
        )


_COMMON = frozenset("a an and are as at be by for from has have in is it of on or that the this to was were with what which how do does".split())


def fake_embed(text: str, size: int = 256) -> list:
    """A deterministic embedding: texts that share words get similar vectors.

    Not a real embedding model. It lets retrieval code be checked with no model and no network.
    """
    vector = [0.0] * size
    for word in re.findall(r"[a-z0-9]+", str(text).lower()):
        if word in _COMMON:
            continue
        word = word[:-1] if len(word) > 3 and word.endswith("s") else word   # so "cats" lands with "cat"
        slot = int(hashlib.md5(word.encode()).hexdigest(), 16) % size
        vector[slot] += 1.0
    length = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / length for v in vector]
