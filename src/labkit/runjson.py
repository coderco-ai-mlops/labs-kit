"""Runs a lab's checks and prints the results as JSON. Used by the labs site on its Linux machine.

    python3 -I -m labkit.runjson <lab folder> <checks file>
"""

import json
import os
import sys
from pathlib import Path


def main() -> None:
    folder, checks_file = Path(sys.argv[1]), Path(sys.argv[2])
    from labkit import checks as registry
    from labkit.checks import CheckFail, Lab
    os.chdir(folder)
    registry._REGISTRY.clear()
    exec(compile(checks_file.read_text(encoding="utf-8"), "checks.py", "exec"), {"__name__": "lab_checks"})
    lab = Lab(folder, sys.argv[3] if len(sys.argv) > 3 else "exercise.py")
    results = []
    for title, hint, fn in registry.registered():
        try:
            fn(lab)
            results.append({"title": title, "ok": True, "message": "", "hint": hint})
        except CheckFail as err:
            results.append({"title": title, "ok": False, "message": str(err), "hint": hint})
        except AssertionError as err:
            results.append({"title": title, "ok": False, "message": str(err) or "This check did not pass.", "hint": hint})
        except Exception as err:
            results.append({"title": title, "ok": False, "message": f"The check itself failed ({type(err).__name__}: {err}). Please report this.", "hint": ""})
    for result in results:
        result["fault"] = result["message"] in lab.faults
    print("@@RESULTS@@" + json.dumps(results))


if __name__ == "__main__":
    main()
