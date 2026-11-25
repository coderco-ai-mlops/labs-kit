"""A practice runner that stands in for dbt on the labs site, where dbt cannot be installed.

It does what dbt does for lab s3-09: it loads the seed CSV into DuckDB (the real DuckDB), resolves
{{ ref('...') }} in each model, builds the models as tables in dependency order, and runs the tests
declared in schema.yml. It reports each step the way dbt prints it. On your own machine the lab's
run_dbt.py uses the real dbt instead.

Included: ref(), {{ config(materialized='table' or 'view') }}, {# comments #}, and the four built-in
tests: unique, not_null, accepted_values and relationships.
"""

import csv
import datetime
import re
from pathlib import Path

import duckdb
import yaml

PROJECT = "shop"
REAL = "It is in the real dbt: see Go further in this lab."
REF = r"ref\(\s*(['\"])([^'\"]+)\1\s*\)"


class Problem(Exception):
    """dbt could not read the project. Nothing is built."""


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    raise AttributeError(f"The practice version of dbt on the labs site does not include `{name}`. {REAL}")


def _line(position: int, total: int, text: str, tag: str) -> None:
    left = f"{position} of {total} {text} "
    print(left + "." * max(3, 78 - len(left)) + f" [{tag}]")


def _summary(counts: dict, details: list) -> None:
    errors = counts["ERROR"]
    print(f"\nCompleted with {errors} error{'' if errors == 1 else 's'}:" if errors else "\nCompleted successfully")
    for detail in details:
        print(f"\n{detail}")
    print(f"\nDone. PASS={counts['PASS']} WARN=0 ERROR={errors} SKIP={counts['SKIP']} TOTAL={sum(counts.values())}")


def _first_line(err: Exception) -> str:
    return "\n  ".join(str(err).strip().splitlines())


# ---- reading the project ---------------------------------------------------

def _compile(name: str, text: str, known: dict) -> tuple:
    """Turn a model's Jinja into plain SQL. Returns the SQL, what it depends on, and how it is stored."""
    depends, kind = [], "table"
    text = re.sub(r"\{#.*?#\}", "", text, flags=re.S)

    def config(match):
        nonlocal kind
        found = re.fullmatch(r"\s*materialized\s*=\s*['\"](table|view)['\"]\s*", match.group(1))
        if not found:
            raise Problem(f"The practice version of dbt on the labs site only includes config(materialized='table') and config(materialized='view'). {name}.sql has: config({match.group(1).strip()}). {REAL}")
        kind = found.group(1)
        return ""

    def ref(match):
        target = match.group(2)
        if target not in known:
            raise Problem(f"Compilation Error\n  Model 'model.{PROJECT}.{name}' (models/{name}.sql) depends on a node named '{target}' which was not found")
        node = f"{known[target]}.{PROJECT}.{target}"
        if node not in depends:
            depends.append(node)
        return f'"main"."{target}"'

    text = re.sub(r"\{\{\s*config\((.*?)\)\s*\}\}", config, text, flags=re.S)
    text = re.sub(r"\{\{\s*" + REF + r"\s*\}\}", ref, text)
    left = re.search(r"\{\{.*?\}\}|\{%.*?%\}|\{\{|\{%", text, flags=re.S)
    if left:
        raise Problem(f"The practice version of dbt on the labs site does not include this piece of Jinja in {name}.sql: {left.group(0)}\n"
                      f"It includes {{{{ ref('...') }}}}, {{{{ config(materialized='...') }}}} and {{# comments #}}. {REAL}")
    return text, depends, kind


def _order(depends: dict) -> list:
    """Model names with every model after the ones it depends on. Ties go in alphabetical order."""
    done, left = [], sorted(depends)
    while left:
        ready = [name for name in left if all(node.split(".")[-1] in done or node.startswith("seed.") for node in depends[name])]
        if not ready:
            # Leave out the models that only wait for the loop, so the message names the loop itself.
            while any(not any(f"model.{PROJECT}.{name}" in depends[other] for other in left) for name in left):
                left = [name for name in left if any(f"model.{PROJECT}.{name}" in depends[other] for other in left)]
            raise Problem("Compilation Error\n  Found a cycle: " + " --> ".join(f"model.{PROJECT}.{name}" for name in left))
        done.append(ready[0])
        left.remove(ready[0])
    return done


def _tests(text: str, known: dict) -> list:
    """The tests in schema.yml, as dicts with kind, model, column, arguments and what they depend on."""
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as err:
        mark = getattr(err, "problem_mark", None)
        where = f"\n    Syntax error near line {mark.line + 1}" if mark else ""
        raise Problem(f"Parsing Error\n  Error reading {PROJECT}: schema.yml - Runtime Error{where}\n    Raw Error:\n    " + "\n    ".join(str(err).splitlines())) from None
    if not isinstance(data, dict):
        raise Problem("Parsing Error\n  The schema file at models/schema.yml is invalid because it is not a set of keys such as `version:` and `models:`")
    if "models" not in data:
        return []
    if not isinstance(data["models"], list):
        raise Problem("Parsing Error\n  The schema file at models/schema.yml is invalid because the value of 'models' is not a list")
    found, seen = [], set()
    for entry in data["models"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise Problem(f"Parsing Error\n  Invalid models config given in models/schema.yml: every item under `models:` needs a `name`. Found: {entry!r}")
        model = entry["name"]
        if model in seen:
            raise Problem(f"Compilation Error\n  dbt found two schema.yml entries for the same resource named {model}. Resources and their associated columns may only be described a single time. To fix this, remove one of the resource entries for {model} in this file:\n   - models/schema.yml")
        seen.add(model)
        if known.get(model) != "model":
            print(f"[WARNING]: Did not find matching node for patch with name '{model}' in the 'models' section of file 'models/schema.yml'")
            continue
        columns = entry.get("columns") or []
        if not isinstance(columns, list) or any(not isinstance(column, dict) or "name" not in column for column in columns):
            raise Problem(f"Parsing Error\n  Invalid models config given in models/schema.yml: `columns` of {model} must be a list of items that each have a `name`")
        for column in [{**entry, "name": None}] + columns:
            listed = (column.get("data_tests") or []) + (column.get("tests") or []) if isinstance(column.get("data_tests") or [], list) and isinstance(column.get("tests") or [], list) else None
            if listed is None:
                raise Problem(f"Parsing Error\n  Invalid test config given in models/schema.yml: `data_tests` of {model} must be a list, with each test on its own line starting with a dash")
            for item in listed:
                kind, arguments = (item, {}) if isinstance(item, str) else next(iter(item.items())) if isinstance(item, dict) and len(item) == 1 else (None, None)
                if kind is None:
                    raise Problem(f"Parsing Error\n  Invalid test config given in models/schema.yml: a test is a name such as `unique`, or a name with its arguments. Found: {item!r}")
                arguments = dict(arguments or {})
                arguments.update(arguments.pop("arguments", None) or {})
                depends = [f"model.{PROJECT}.{model}"]
                to = re.fullmatch(REF, str(arguments.get("to", "")).strip())
                if kind == "relationships" and to:
                    if to.group(2) not in known:
                        raise Problem(f"Compilation Error\n  Test on '{model}' (models/schema.yml) depends on a node named '{to.group(2)}' which was not found")
                    arguments["to"] = to.group(2)
                    depends.insert(0, f"{known[to.group(2)]}.{PROJECT}.{to.group(2)}")
                suffix = "__".join(str(value) for value in arguments.get("values", [])) if kind == "accepted_values" else ""
                found.append({"kind": kind, "model": model, "column": column["name"], "arguments": arguments, "depends": tuple(depends),
                              "name": "_".join(part for part in (kind, model, column["name"], suffix and "_" + suffix) if part)})
    return sorted(found, key=lambda test: test["name"])


# ---- running it ------------------------------------------------------------

def _column_type(values: list) -> tuple:
    """The DuckDB type for one CSV column, and the function that converts its text."""
    filled = [value for value in values if value != ""]
    for kind, convert in (("integer", int), ("double", float), ("date", datetime.date.fromisoformat)):
        try:
            for value in filled:
                convert(value)
            return kind, convert
        except ValueError:
            continue
    return "varchar", str


def _seed(connection, path: Path) -> int:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, rows = rows[0], rows[1:]
    kinds = [_column_type([row[i] for row in rows]) for i in range(len(header))]
    connection.execute(f'create or replace table "main"."{path.stem}" (' + ", ".join(f'"{name}" {kind}' for name, (kind, _) in zip(header, kinds)) + ")")
    for row in rows:
        values = [None if value == "" else convert(value) for value, (_, convert) in zip(row, kinds)]
        connection.execute(f'insert into "main"."{path.stem}" values (' + ", ".join("?" for _ in values) + ")", values)
    return len(rows)


def _failures(connection, test: dict) -> int:
    """How many rows fail a test. The SQL is the same as dbt's own."""
    kind, column, arguments, table = test["kind"], test["column"], test["arguments"], f'"main"."{test["model"]}"'
    if kind == "not_null":
        sql, values = f"select count(*) from {table} where {column} is null", []
    elif kind == "unique":
        sql, values = f"select count(*) from (select {column} from {table} where {column} is not null group by {column} having count(*) > 1)", []
    elif kind == "accepted_values":
        values = [str(value) for value in arguments["values"]]
        sql = f"select count(*) from (select distinct {column} as value_field from {table}) where value_field::varchar not in ({', '.join('?' for _ in values)})"
    elif kind == "relationships":
        sql, values = (f'select count(*) from (select {column} as from_field from {table} where {column} is not null) child '
                       f'left join (select {arguments["field"]} as to_field from "main"."{arguments["to"]}") parent on child.from_field = parent.to_field where parent.to_field is null'), []
    else:
        raise LookupError(f"'test_{kind}' is undefined. dbt has no test called `{kind}`. The built-in tests are unique, not_null, accepted_values and relationships.")
    return connection.execute(sql, values).fetchone()[0]


def build(project: dict, folder, seed_file: str, files: list) -> dict:
    """Run dbt seed, dbt run and dbt test on the lab's files, filling in `project` for the lab's checks."""
    folder = Path(folder)
    seed = Path(seed_file).stem
    sources = {Path(name).stem: (folder / name).read_text(encoding="utf-8") for name in files if name.endswith(".sql")}
    known = {seed: "seed", **{name: "model" for name in sources}}
    print("Running with the practice version of dbt")
    try:
        compiled = {name: _compile(name, text, known) for name, text in sources.items()}
        order = _order({name: depends for name, (_, depends, _) in compiled.items()})
        tests = []
        for name in files:
            if name.endswith((".yml", ".yaml")):
                tests += _tests((folder / name).read_text(encoding="utf-8"), known)
    except Problem as err:
        project["problem"] = f"dbt could not read the project:\n{err}"
        return project
    found = f"Found {len(order)} model{'' if len(order) == 1 else 's'}, 1 seed, {len(tests)} data test{'' if len(tests) == 1 else 's'}\n"
    connection = duckdb.connect(":memory:")

    print(f"\n$ dbt seed\n{found}")
    _line(1, 1, f"START seed file main.{seed}", "RUN")
    try:
        _line(1, 1, f"OK loaded seed file main.{seed}", f"INSERT {_seed(connection, folder / seed_file)}")
        _summary({"PASS": 1, "ERROR": 0, "SKIP": 0}, [])
    except Exception as err:
        _line(1, 1, f"ERROR loading seed file main.{seed}", "ERROR")
        _summary({"PASS": 0, "ERROR": 1, "SKIP": 0}, [f"[ERROR]: in seed {seed} ({seed_file})\n  {_first_line(err)}"])
        project["problem"] = f"dbt could not load the seed data. If you changed {seed_file}, put it back as it was."
        return project

    print(f"\n$ dbt run\n{found}")
    counts, details = {"PASS": 0, "ERROR": 0, "SKIP": 0}, []
    for position, name in enumerate(order, 1):
        sql, depends, kind = compiled[name]
        project["depends"][name] = depends
        if any(project["models"].get(node.split(".")[-1], ("success",))[0] != "success" for node in depends):
            project["models"][name] = ("skipped", "")
            counts["SKIP"] += 1
            _line(position, len(order), f"SKIP relation main.{name}", "SKIP")
            continue
        _line(position, len(order), f"START sql {kind} model main.{name}", "RUN")
        try:
            connection.execute(f'drop view if exists "main"."{name}"' if kind == "table" else f'drop table if exists "main"."{name}"')
            connection.execute(f'create or replace {kind} "main"."{name}" as (\n{sql}\n)')
            project["models"][name] = ("success", "OK")
            counts["PASS"] += 1
            _line(position, len(order), f"OK created sql {kind} model main.{name}", "OK")
        except duckdb.Error as err:
            message = f"Runtime Error in model {name} (models/{name}.sql)\n  {_first_line(err)}"
            project["models"][name] = ("error", message)
            counts["ERROR"] += 1
            details.append(f"[ERROR]: in model {name} (models/{name}.sql)\n  {message}")
            _line(position, len(order), f"ERROR creating sql {kind} model main.{name}", "ERROR")
    if order:
        _summary(counts, details)
    else:
        print("Nothing to do. No model has any SQL yet.")

    print(f"\n$ dbt test\n{found}")
    counts, details = {"PASS": 0, "ERROR": 0, "SKIP": 0}, []
    for position, test in enumerate(tests, 1):
        key = (test["kind"], test["column"], test["depends"])
        project["tests"].append(key)
        _line(position, len(tests), f"START test {test['name']}", "RUN")
        try:
            failures = _failures(connection, test)
        except Exception as err:
            project["test_results"][key] = ("error", None)
            counts["ERROR"] += 1
            details.append(f"[ERROR]: in test {test['name']} (models/schema.yml)\n  Runtime Error in test {test['name']} (models/schema.yml)\n  {_first_line(err)}")
            _line(position, len(tests), f"ERROR {test['name']}", "ERROR")
            continue
        project["test_results"][key] = ("fail" if failures else "pass", failures)
        counts["ERROR" if failures else "PASS"] += 1
        if failures:
            details.append(f"[ERROR]: in test {test['name']} (models/schema.yml)\n  Got {failures} result{'' if failures == 1 else 's'}, configured to fail if != 0")
        _line(position, len(tests), f"FAIL {failures} {test['name']}" if failures else f"PASS {test['name']}", f"FAIL {failures}" if failures else "PASS")
    if tests:
        _summary(counts, details)
    else:
        print("Nothing to do. schema.yml declares no tests for these models.")

    for name, (status, _) in project["models"].items():
        if status == "success":
            result = connection.execute(f'select * from "main"."{name}" order by 1')
            columns = [column[0].lower() for column in result.description]
            project["tables"][name] = [dict(zip(columns, row)) for row in result.fetchall()]
    connection.close()
    return project
