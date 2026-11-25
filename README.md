<img src="assets/banner.svg" alt="CoderCo AI & MLOps: hands-on labs" width="100%">

# labs-kit

The shared tooling behind every CoderCo AI & MLOps lab repo. It holds the `lab` command and the helpers that lab checks are written with. It holds no course content.

## The lab command

| Command | What it does |
|---|---|
| `lab doctor` | Checks the machine is ready. |
| `lab list` | Shows every lab, its time estimate and your progress. |
| `lab start <id>` | Shows a lab's task card. |
| `lab check <id>` | Checks your work. `--all` checks every lab. |
| `lab hint <id>` | Shows the next hint. |
| `lab reset <id>` | Puts a lab's files back to the start. |
| `lab verify --solutions <folder>` | For maintainers: proves every starter fails and every solution passes. |

## A lab folder

```
labs/s1-05-lists/
  README.md     the task card
  lab.toml      id, title, section, minutes, files, checks
  exercise.py   the starter the member edits
  checks.py     the checks
  hints.md      three hints, each under a "## " heading
  .starter/     the untouched starter, used by lab reset
```

## Writing checks

```python
from labkit import CheckFail, check

@check("population holds a number")
def _(lab):
    lab.script().need_number("population")

@check("Prints adult from 18 up")
def _(lab):
    s = lab.script(set={"age": 18})   # rerun the member's file with a different value
    s.need_line("adult")
```

Every message a check raises is shown to the member as written, so write it for a beginner. labkit uses only the Python standard library.
