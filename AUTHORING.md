# Writing a lab

The standard every CoderCo AI & MLOps lab follows. The 12 labs in `python-labs/labs/s1-*` are the reference: read two of them before writing one (`s1-08-if-statements` and `s1-11-functions` show most of the techniques).

## What a lab is

One lesson page becomes one lab. A member reads the task, writes code, and one command says pass or fail with a message that tells them what to fix. A lab is finished only when the proof passes: the untouched starter fails its checks and the solution passes every one.

## The folder

```
labs/s2-03-error-handling/
  README.md      the task card (format below)
  lab.toml       settings (below)
  exercise.py    the starter the member edits. More than one file is allowed.
  checks.py      the checks
  hints.md       exactly three hints
  .starter/      a byte-for-byte copy of every starter file, used by `lab reset`
```

The solution goes in `labs-solutions/python/<same folder name>/`, one file per starter file, same names. Nothing else goes in the solution folder.

## lab.toml

```toml
id = "s2-03"
title = "Error handling"
section = "Section 2: Python for Production"
minutes = 20
files = ["exercise.py"]
checks = "checks.py"
where = "browser"        # or "terminal"
packages = []            # third-party imports, e.g. ["numpy", "pandas"]
group = ""               # dependency group that installs them, e.g. "data"
timeout = 10             # seconds; raise it only when a lab truly needs longer
```

Section names, exactly:
- `Section 2: Python for Production`
- `Section 3: Python for Data Engineering`
- `Section 4: Python for ML Engineering`
- `Section 5: Python for GenAI Engineering`

`minutes` is an honest estimate for a learner at that point in the course. Lesson exercises are usually 15 to 30. Mini-projects are 90 to 150.

## Every lab runs on the labs site

Members do labs on the labs site, where Python runs inside the browser. Every lab must run there. The same lab also runs in a terminal with the real tools, and the same checks and the same solution must pass in both.

A lab gets its tools in one of two ways.

**A real package.** List it in `packages`. The site installs it when the lab opens. Known to work: `numpy`, `pandas`, `scipy`, `scikit-learn`, `joblib`, `fastapi`, `pydantic`, `httpx`, `openai`, `langfuse`, `great-expectations`, `duckdb`, `pyyaml`, `pyarrow`, `pytest`, `prometheus-client`, `mlflow-skinny` (with `sqlalchemy` and `alembic`, on a SQLite store). Prefer this whenever it works.

**A practice version.** Some tools cannot be installed in a browser: PyTorch, TensorFlow, Airflow, dbt, Chroma, LangChain, LangGraph, Feast, Evidently, BentoML. For those there is a small stand-in under `labs-kit/src/labkit/standins/`, listed in the lab's `standins`. It behaves the same as the real tool for everything the lab uses, and says so plainly when a member reaches for something it does not include. See `standins/README.md`.

What the browser cannot do: open network connections, start a server, use threads or subprocesses. So an API lab calls its app through `labkit.web.call(app, "GET", "/health")` in place of a running server, and a model lab passes in a `FakeLLM`.

Things with no Python tool to run, such as a `docker-compose.yml`, a GitHub Actions workflow or a Grafana dashboard, are configuration files the member edits and the checks parse strictly. List them in `files`.

A lab can also ship files the member does not edit. Small data and config files (`.csv`, `.json`, `.txt`, `.yml`, `.yaml`, `.toml`, `.sql`) in the lab folder travel to the site automatically. Name anything else, such as a helper module, in `support = ["helper.py"]`. Binary files do not travel: create them in code.

On the site, a lab that needs a library the current Python does not have gets a fresh Python, so a lab never depends on what was opened before it.

Each module is its own repo (`python-labs`, `mlops-labs`) with its solutions under `labs-solutions/<module>/`. For a module other than Python, pass `--module=mlops` to the browser check.

`where = "terminal"` still exists for a lab that truly cannot run on the site. No Python lab needs it today. Do not use it without proving the two routes above cannot work.

Prove a lab three ways before calling it done:

```
cd labs-platform && node tools/browser-check.mjs <lab id>                 # in the browser's Python
cd python-labs && uv run lab verify --solutions ../labs-solutions/python <lab id>     # with the real tool
cd python-labs && LAB_STANDINS=1 uv run lab verify --solutions ../labs-solutions/python <lab id>   # with the practice version, if it has one
```

On the task card, never tell a member that something cannot run in a web page. If a lab uses a practice version, say so in one short paragraph at the end of "Build it", and put the real-tool route in "Go further".

## Linux labs (Engineering Foundations)

A lab with `where = "linux"` runs on a real Linux machine in the member's browser: Alpine Linux 3.21 (32-bit), run by the v86 emulator, with a terminal on the right of the page. The member is `learner` (home `/home/learner`, passwordless `sudo`) and starts in `~/lab`. There is no network to the internet inside the machine; everything a lab needs is already installed. See `labs-platform/linux/Dockerfile` for the full package list. It includes bash, coreutils, findutils, grep, sed, gawk, procps, psmisc, shadow, sudo, nano, vim, tree, less, tar, gzip, zip, git, curl, wget, jq, yq, sqlite, redis, Postgres 16, python3 with PyYAML, openssh-client, openssl, iproute2, iputils, bind-tools, netcat, nginx, dnsmasq and iptables.

A Linux lab folder:

```
labs/ef3-02-permissions/
  README.md     the task card. Commands are typed in the terminal, so write them as the member types them.
  lab.toml      where = "linux"; files = the files the member edits (may be empty)
  setup.sh      optional, runs as root before the member starts. $LAB_DIR is /home/learner/lab.
  checks.py     runs as root on the same machine, after the member presses Check
  hints.md      three hints
  ...           any other text files: copied into ~/lab as they are
  .starter/     a copy of the files listed in `files`
```

The solution goes in `labs-solutions/foundations/<folder>/`: edited files under their names, plus `solution.sh`, which is run as `learner` from `~/lab` and does what a member would type. Both are optional, but at least one must exist.

Checks use `labkit.shell` (read `labs-kit/src/labkit/shell.py`): `sh(cmd)` runs a command as the member and returns `.code`, `.out`, `.err`, `.lines`; `need_file`, `need_dir`, `read`, `mode` (for example `"750"`), `owner`, `group`, `user_exists`, `groups_of`, `processes`, `running`, `git(args, cwd)`. Check outcomes on the machine, not what the member typed. `~/.bash_history` is written after every command, but use it only when the lesson is the command itself.

The member's whole home folder is saved to their account and comes back next visit. Anything outside it (users added, services started, files in `/etc`) starts fresh each visit, so a lab must be finishable in one sitting, or keep its state under `/home/learner`. `setup.sh` should make everything the lab needs outside the home folder.

Prove a Linux lab with `cd labs-platform && python3 build.py && node tools/linux-check.mjs <lab id>`. It boots the same machine members get, checks the untouched starter (some checks must fail), applies the solution and checks again (all must pass). Then try two wrong answers a member would give, by hand, with a small script of your own that uses `site/linuxlab.mjs`, and read the messages.

On a Linux lab card, "Before you start" says the terminal on the right is a real Linux machine. "Check" says "Press Check." Do not mention `uv` or `lab check`.

## The task card (README.md)

Copy the shape of `s1-08-if-statements/README.md` exactly: same headings, same order.

1. `# <id>  <Title>`
2. `**Outcome:**` one sentence: what they will have built.
3. `**Time:**` about N minutes
4. `**Lesson:**` the page's title
5. `## Before you start` (the `uv run lab doctor` block; for a terminal lab add the install command, e.g. `uv sync --group data`)
6. `## Build it` numbered steps. Name every variable, function, class, file and exact output format the checks rely on. Then how to run the file, then example output.
7. `## Break it on purpose` one deliberate failure, what they will see, and how to put it right.
8. `## Check` the `uv run lab check <id>` block.
9. `## Pitfalls` a two-column table: `You see` and `What to do`. Real error strings.
10. `## Go further` optional, unchecked.

Base the steps on the page's own exercise. Where the page leaves something open ("print a message"), pin it down, because a check needs something exact. Where the page asks for something that cannot be checked ("explain to yourself"), keep it as a step and do not check it.

## Voice

- Plain, conversational English in full sentences. Written for someone who is new to this.
- No em dashes and no en dashes. Use "to" for ranges.
- No idioms, no hype, no jokes.
- British spelling in prose (colour, organise). Code uses whatever the library uses.
- Never mention a paid service as required. If a page uses a paid API or a card-gated cloud account, the lab uses the free route (below) and names the paid one as optional.

## Starter files

Comments that mirror the steps, with gaps. No working answer in the starter. Start with:

```python
# Lab s2-03: Error handling
# Steps are in README.md. Run this file with:
#   uv run python labs/s2-03-error-handling/exercise.py
```

Keep those three comment lines in exactly that form: the labs site removes them automatically.

## Checks (checks.py)

```python
"""Checks for lab s2-03. Run them with: uv run lab check s2-03"""

from labkit import CheckFail, check


@check("safe_divide returns None when dividing by zero")
def _(lab):
    fn = lab.script().need_function("safe_divide")
    got = lab.call(fn, 10, 0)
    if got is not None:
        raise CheckFail(f"`safe_divide(10, 0)` should return None. It returned {got!r}.")
```

What `lab` gives you (read `labs-kit/src/labkit/checks.py`, it is short):

| Call | Use |
|---|---|
| `lab.script(name=None)` | Runs a member file once and caches it. Returns a result with `.vars`, `.out`, `.lines`, `.error`. |
| `lab.script(set={"age": 17})` | Reruns the file with a top-level variable replaced. Catches hard-coded answers. |
| `s.need("x", type)` / `s.need_number("x")` / `s.need_function("f")` | Fetch a name or fail with a clear message. |
| `s.need_line(text)` / `s.need_printed(text)` | The output has this exact line / contains this text. |
| `lab.call(fn, *args)` | Calls a member function, hides its prints, and turns an exception into a clear message. |
| `lab.source(name=None)` | The file's text, for the rare check on how something was written. |
| `lab.path(name)` | Path to a file in the lab folder. |

For classes: fetch the class with `s.need("BankAccount", type)`, build instances yourself and test behaviour.

For labs that call a model, use `labkit.llm` (`FakeLLM`, `fake_embed`). See "Model labs" below.

Rules for checks:
- **Test behaviour, not text.** Call the function with several inputs, including edges. Only inspect source when the lesson is about how it is written (for example "use a loop").
- **3 to 8 checks per lab.** Each has a title a learner understands, describing what should be true.
- **Every failure message says what was expected and what happened**, with the real values, in words a beginner can act on. Never show a raw traceback.
- **Deterministic.** Fix every random seed. No clock, no network, no dependence on dictionary or set order. For model metrics, check a generous floor, never an exact number.
- **The member's values are theirs.** Where they choose their own data, check relationships (the total equals the sum of their list), not fixed answers.
- **The starter must fail every check** where possible, and must not pass them all.
- Checks import only the standard library, `labkit`, and the lab's declared `packages`.

## Hints (hints.md)

```
# Hints for s2-03

## Hint 1

A nudge: which idea from the page applies.

## Hint 2

The shape of the answer, without the answer.

## Hint 3

Nearly the answer, for the hardest step.
```

## Model labs (section 5)

Checks never call a real model. Code under test takes the client as an argument:

```python
def ask(client, question: str, model: str = "llama3.2") -> str:
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": question}])
    return response.choices[0].message.content
```

The check passes `FakeLLM([...])` and inspects `fake.calls`. This is the OpenAI-compatible shape, which Ollama and the free hosted tiers all accept, so the same function runs against a real model unchanged.

Keep `exercise.py` free of imports that do not exist in the browser (no `import openai` at the top). Put the real run in a second file, `live.py`, which is not listed in `files` and is never checked: it builds a real client and calls the member's functions. Its default is Ollama on the laptop (`base_url="http://localhost:11434/v1"`), with a free hosted tier as the alternative and a paid key as optional.

For embeddings and retrieval, check with `fake_embed`. Teach the real embedding model and the real vector database in `live.py` or in a terminal lab.

## Proof

From `python-labs/`, using the Python environment you were given:

```
<python> -m labkit.cli verify --solutions ../labs-solutions/python <lab id>
```

A lab is done when it prints PASS with "starter fails N of N, solution passes N". Then try three wrong answers a real learner would write and read the messages they get. Fix any message that would not help them.
