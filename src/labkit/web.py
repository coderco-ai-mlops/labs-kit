"""Send requests to a web app (FastAPI, Starlette or any ASGI app) without starting a server.

    from labkit.web import call

    response = call(app, "POST", "/order", json={"product": "keyboard", "quantity": 2, "price": 45.5})
    print(response.status, response.body)      # 200 {'product': 'keyboard', 'total': 91.0}

`call` hands the request straight to the app, the same way a server such as uvicorn would,
and gives back what the app answered. Nothing listens on a port, so it works on the labs
site and in a terminal alike. A lab's checks use it, and so does a lab file's own
`if __name__ == "__main__":` block, so that running the file shows real requests and responses.

    with running(app) as api:                   # runs the app's startup code, then its shutdown code
        print(api.call("GET", "/ready").status)

`call` on its own does not run startup code. `running` does, once, like a real server.
Inside async code, use `await request(app, ...)`.
"""

import asyncio
import json as _json
import re
import sys
from asyncio import events
from urllib.parse import urlencode

__all__ = ["call", "request", "running", "Response", "FakeAsyncLLM"]

_BROWSER = sys.platform == "emscripten"


class Response:
    """What the app answered.

    status   the status code, for example 200
    headers  a dictionary, with the names in lower case
    body     the answer: decoded from JSON when it is JSON, otherwise the text
    text     the answer as text, exactly as it was sent
    chunks   the pieces the answer was sent in, in order (a streamed answer has several)
    error    the exception the endpoint raised, or None. An endpoint that raises gives status 500.
    """

    def __init__(self):
        self.status = None
        self.headers = {}
        self.content = b""
        self.chunks = []
        self.error = None

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self):
        return _json.loads(self.text)

    @property
    def body(self):
        if "json" in self.headers.get("content-type", ""):
            try:
                return self.json()
            except ValueError:
                pass
        return self.text

    def __repr__(self):
        return f"<Response {self.status} {self.body!r}>"


async def request(app, method: str, path: str, *, json=None, content=None, headers=None, params=None,
                  overrides=None, on_chunk=None, state=None) -> Response:
    """Send one request to the app and return its Response. This is the async form of `call`."""
    if not callable(app):
        raise TypeError(f"The first argument should be your app, for example call(app, \"GET\", \"/health\"). It was a {type(app).__name__}.")
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError(f"The path should be text that starts with a slash, such as \"/health\". It was {path!r}.")
    method = str(method).upper()
    path, _, query = path.partition("?")
    if params:
        query = "&".join(part for part in (query, urlencode(params, doseq=True)) if part)
    sent = {"host": "labs.local"}
    payload = b""
    if json is not None:
        payload = _json.dumps(json).encode("utf-8")
        sent["content-type"] = "application/json"
    elif content is not None:
        payload = content.encode("utf-8") if isinstance(content, str) else bytes(content)
    if payload:
        sent["content-length"] = str(len(payload))
    sent.update({str(key).lower(): str(value) for key, value in (headers or {}).items()})
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": path, "raw_path": path.encode("utf-8"),
        "query_string": query.encode("utf-8"), "root_path": "",
        "headers": [(key.encode("latin-1"), value.encode("latin-1")) for key, value in sent.items()],
        "client": ("labs-client", 50000), "server": ("labs.local", 80),
    }
    if state is not None:
        scope["state"] = dict(state)

    response = Response()
    finished = asyncio.Event()
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await finished.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            response.status = message["status"]
            response.headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in message.get("headers", [])}
        elif message["type"] == "http.response.body":
            piece = message.get("body", b"")
            if piece:
                response.content += piece
                text = piece.decode("utf-8", errors="replace")
                response.chunks.append(text)
                if on_chunk is not None:
                    on_chunk(text)
            if not message.get("more_body", False):
                finished.set()

    _use_no_threads()
    saved = None
    if overrides:
        if not hasattr(app, "dependency_overrides"):
            raise TypeError("`overrides` replaces FastAPI dependencies, so the app must be a FastAPI app.")
        saved = dict(app.dependency_overrides)
        app.dependency_overrides.update(overrides)
    try:
        await app(scope, receive, send)
    except Exception as err:   # a server answers 500 when an endpoint raises, and so does this
        response.error = err
        if response.status is None:
            response.status = 500
            response.headers = {"content-type": "text/plain; charset=utf-8"}
            response.content = b"Internal Server Error"
            response.chunks = ["Internal Server Error"]
    finally:
        finished.set()
        if saved is not None:
            app.dependency_overrides.clear()
            app.dependency_overrides.update(saved)
    if response.status is None:
        response.status = 500
        response.error = RuntimeError("The app finished without sending a response.")
    return response


def call(app, method: str, path: str, **options) -> Response:
    """Send one request to the app and return its Response.

    call(app, "GET", "/health")
    call(app, "POST", "/order", json={"product": "keyboard", "quantity": 2, "price": 45.5})
    call(app, "GET", "/search", params={"q": "desk"}, headers={"x-token": "abc"})

    `overrides={get_client: lambda: fake}` replaces FastAPI dependencies for this one request.
    `on_chunk=function` is called with each piece of a streamed answer as it is sent.
    The app's startup code does not run. For that, use `with running(app) as api:`.
    """
    runner = _Runner()
    try:
        return runner.run(request(app, method, path, **options))
    finally:
        runner.close()


class running:
    """Start the app the way a server does, send it requests, then shut it down.

        with running(app) as api:
            response = api.call("GET", "/ready")

    Entering the block runs the app's startup code (a lifespan function, or startup handlers).
    Leaving it runs the shutdown code. An error in the startup code is raised here.
    """

    def __init__(self, app, overrides=None):
        if not callable(app):
            raise TypeError(f"running() needs your app, for example `with running(app) as api:`. It was given a {type(app).__name__}.")
        self.app = app
        self.overrides = dict(overrides or {})
        self.state = {}
        self._runner = None
        self._task = None

    def __enter__(self):
        self._runner = _Runner()
        try:
            self._runner.run(self._startup())
        except BaseException:
            self._runner.close()
            self._runner = None
            raise
        return self

    def __exit__(self, *exc):
        runner, self._runner = self._runner, None
        try:
            runner.run(self._shutdown())
        finally:
            runner.close()
        return False

    def call(self, method: str, path: str, **options) -> Response:
        if self._runner is None:
            raise RuntimeError("The app is not running. Use `with running(app) as api:` and call `api.call(...)` inside the block.")
        options["overrides"] = {**self.overrides, **(options.get("overrides") or {})}
        return self._runner.run(request(self.app, method, path, state=self.state, **options))

    async def _startup(self):
        _use_no_threads()
        self._inbox, self._outbox = asyncio.Queue(), asyncio.Queue()

        async def main():
            scope = {"type": "lifespan", "asgi": {"version": "3.0", "spec_version": "2.0"}, "state": self.state}
            try:
                await self.app(scope, self._inbox.get, self._outbox.put)
            except BaseException as err:
                await self._outbox.put({"type": "ended", "error": err})
            else:
                await self._outbox.put({"type": "ended", "error": None})

        self._task = asyncio.ensure_future(main())
        await self._inbox.put({"type": "lifespan.startup"})
        await self._wait_for("lifespan.startup.complete", "start")

    async def _shutdown(self):
        if self._task is None or self._task.done():
            return
        await self._inbox.put({"type": "lifespan.shutdown"})
        await self._wait_for("lifespan.shutdown.complete", "shut down")

    async def _wait_for(self, wanted: str, doing: str):
        failure = None
        while True:
            message = await self._outbox.get()
            if message["type"] == wanted:
                return
            if message["type"] == "ended":
                # An app with no startup code at all ends here, or raises without reporting a failure.
                # A server carries on in that case, and so does this.
                await self._task
                if message["error"] is not None and (failure is not None or doing != "start"):
                    raise message["error"]
                if failure is not None:
                    raise RuntimeError(f"The app failed to {doing}. {failure}")
                return
            if message["type"].endswith(".failed"):
                failure = str(message.get("message", "")).strip().splitlines()[-1:] or [""]
                failure = failure[0]


# Running async code from ordinary code.
#
# In a terminal this is a normal asyncio event loop. A browser has no threads and its own event loop
# counts as already running, so there the requests run on a small loop that only knows how to wait.

class _Wait:
    def select(self, timeout):
        if timeout is None:
            raise RuntimeError("The app is waiting for something that never happens.")
        if timeout > 0:
            import time

            time.sleep(timeout)
        return []


class _SmallLoop(asyncio.BaseEventLoop):
    def __init__(self):
        super().__init__()
        self._selector = _Wait()

    def _process_events(self, event_list):
        pass

    def _write_to_self(self):
        pass


class _Runner:
    """Runs coroutines one after another on one event loop, so an app can stay started between requests."""

    def __init__(self):
        self.loop = None

    def run(self, coroutine):
        outer = events._get_running_loop()
        if outer is not None and (not _BROWSER or isinstance(outer, asyncio.BaseEventLoop)):
            coroutine.close()
            raise RuntimeError("call() was used inside async code. There, use `response = await request(app, ...)` from labkit.web.")
        if self.loop is None:
            self.loop = _SmallLoop() if _BROWSER else asyncio.new_event_loop()
        events._set_running_loop(None)
        try:
            return self.loop.run_until_complete(coroutine)
        finally:
            events._set_running_loop(outer)

    def close(self):
        loop, self.loop = self.loop, None
        if loop is None or loop.is_closed():
            return
        outer = events._get_running_loop()
        events._set_running_loop(None)
        try:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            if not _BROWSER:
                loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            loop.close()
            events._set_running_loop(outer)


_no_threads_done = False


def _use_no_threads():
    """FastAPI runs an ordinary `def` endpoint in a worker thread. A browser has no threads, so there it runs in place."""
    global _no_threads_done
    if not _BROWSER or _no_threads_done:
        return
    try:
        import anyio.to_thread
    except ImportError:
        return

    async def run_sync(func, *args, **kwargs):
        return func(*args)

    anyio.to_thread.run_sync = run_sync
    _no_threads_done = True


# A model client for web apps, for checks and for a lab file's own example requests.

class _Obj:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def __repr__(self):
        return f"_Obj({self.__dict__})"


class FakeAsyncLLM:
    """Looks like openai.AsyncOpenAI: `await client.chat.completions.create(model=..., messages=[...])`.

    With `stream=True` the call gives an async iterator of chunks, each with `choices[0].delta.content`.
    As with a real model, the first chunk's content is empty text and the last chunk's content is None.
    `replies` are used in order, and the last one repeats. A reply that is an Exception is raised, to practise retries.
    Every call's keyword arguments are kept in `.calls`. `.events` records ("made", piece) each time
    a piece of a streamed reply is produced.

        fake = FakeAsyncLLM(["An API lets programs talk to each other."])
        call(app, "POST", "/chat", json={"message": "What is an API?"}, overrides={get_client: lambda: fake})
    """

    def __init__(self, replies=("ok",)):
        self.replies = list(replies) or ["ok"]
        self.calls: list = []
        self.events: list = []
        self.chat = _Obj(completions=_Obj(create=self._create))

    async def _create(self, **kwargs):
        if "model" not in kwargs or "messages" not in kwargs:
            raise TypeError("create() needs model=... and messages=[...]")
        self.calls.append(kwargs)
        reply = self.replies[min(len(self.calls), len(self.replies)) - 1]
        if isinstance(reply, BaseException):
            raise reply
        if kwargs.get("stream"):
            return self._stream(reply)
        return _Obj(model=kwargs["model"], choices=[_Obj(index=0, finish_reason="stop", message=_Obj(role="assistant", content=reply))])

    async def _stream(self, reply):
        yield _Obj(choices=[_Obj(index=0, finish_reason=None, delta=_Obj(role="assistant", content=""))])
        for piece in re.findall(r"\S+\s*", reply):
            self.events.append(("made", piece))
            yield _Obj(choices=[_Obj(index=0, finish_reason=None, delta=_Obj(content=piece))])
        yield _Obj(choices=[_Obj(index=0, finish_reason="stop", delta=_Obj(content=None))])
