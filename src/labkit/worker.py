"""Runs one lab's checks in its own process and prints the results as JSON.

Kept separate from the CLI so a member's endless loop can be stopped by a timeout.
Usage: python -m labkit.worker <lab folder> <checks file> <default file>
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

from labkit import checks as registry
from labkit.checks import CheckFail, Lab

MARK = "@@labkit-results@@"


def run(folder: Path, checks_file: str, default_file: str) -> list:
    spec = importlib.util.spec_from_file_location("lab_checks", folder / checks_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lab = Lab(folder, default_file)
    results = []
    for title, hint, fn in registry.registered():
        try:
            fn(lab)
            results.append({"title": title, "ok": True, "message": "", "hint": hint})
        except CheckFail as err:
            results.append({"title": title, "ok": False, "message": str(err), "hint": hint})
        except AssertionError as err:
            results.append({"title": title, "ok": False, "message": str(err) or "This check did not pass.", "hint": hint})
        except Exception as err:  # a fault in the check itself must not look like the member's mistake
            results.append({"title": title, "ok": False, "message": f"The check itself failed ({type(err).__name__}: {err}). Please report this.", "hint": ""})
    for result in results:
        result["fault"] = result["message"] in lab.faults
    return results


def main() -> None:
    folder, checks_file, default_file = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    sys.path.insert(0, str(folder))
    if os.environ.get("LAB_STANDINS") == "1":
        # For lab authors: use the practice versions the labs site uses, to prove a lab passes with them.
        sys.path.insert(0, str(Path(__file__).resolve().parent / "standins"))
    results = run(folder, checks_file, default_file)
    sys.stdout.write("\n" + MARK + json.dumps(results) + "\n")


if __name__ == "__main__":
    main()
