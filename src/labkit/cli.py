"""The `lab` command."""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from labkit import core, ui


def _print_results(lab: core.LabInfo, results: list) -> None:
    ui.heading(f"{lab.id}  {lab.title}")
    faults = list(dict.fromkeys(r["message"] for r in results if r.get("fault")))
    if faults:
        # A file that will not run stops several checks at once, so say it once.
        for message in faults:
            print(f"  {ui.coral('FAIL')}  Fix this first")
            for line in message.splitlines():
                print(f"        {line}")
        results = [r for r in results if not r.get("fault")]
        if not results:
            print()
            print(f"  Then run: lab check {lab.id}")
            return
    for result in results:
        if result["ok"]:
            print(f"  PASS  {result['title']}")
            continue
        print(f"  {ui.coral('FAIL')}  {result['title']}")
        for line in result["message"].splitlines():
            print(f"        {line}")
        if result["hint"]:
            print(ui.dim(f"        Hint: {result['hint']}"))
    done = sum(r["ok"] for r in results)
    print()
    if faults:
        print(f"  Fix the first item, then run: lab check {lab.id}")
    elif core.passed(results):
        print(ui.bold(f"  All {len(results)} checks pass. {lab.id} is done."))
    else:
        print(f"  {done} of {len(results)} checks pass. Stuck? Run: lab hint {lab.id}")


def cmd_check(args) -> int:
    root = core.find_root()
    labs = core.all_labs(root) if args.all else [core.find(root, args.lab)] if args.lab else None
    if labs is None:
        raise SystemExit("Say which lab to check, for example: lab check s1-01   (or: lab check --all)")
    report = []
    for lab in labs:
        results = core.run_checks(lab)
        core.record(root, lab, results)
        report.append((lab, results))
        if not args.all:
            _print_results(lab, results)
    if args.all:
        _print_progress(root)
    if args.summary:
        _write_summary(report)
    if args.json:
        print(json.dumps({lab.id: results for lab, results in report}))
    everything = all(core.passed(results) for _, results in report)
    return 0 if everything or not args.strict else 1


def _write_summary(report: list) -> None:
    """A table for the GitHub Actions run page."""
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    done = sum(core.passed(results) for _, results in report)
    lines = [f"## Lab progress: {done} of {len(report)} done", "", "| Lab | Checks | Status |", "|---|---|---|"]
    for lab, results in report:
        ok = sum(r["ok"] for r in results)
        status = "Done" if core.passed(results) else "In progress" if ok else "Not started"
        lines.append(f"| {lab.id} {lab.title} | {ok} of {len(results)} | {status} |")
    text = "\n".join(lines) + "\n"
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text)


def _print_progress(root: Path) -> None:
    progress = core.read_progress(root)
    labs = core.all_labs(root)
    section = None
    for lab in labs:
        if lab.section != section:
            section = lab.section
            ui.heading(section or "Labs")
        entry = progress.get(lab.id, {})
        if entry.get("done"):
            state = "done"
        elif entry.get("passed"):
            state = ui.coral(f"{entry['passed']} of {entry['total']}")
        else:
            state = ui.dim("to do")
        print(f"  {lab.id:<8} {lab.title:<34} {lab.minutes:>3} min   {state}")
    done = sum(1 for lab in labs if progress.get(lab.id, {}).get("done"))
    left = sum(lab.minutes for lab in labs if not progress.get(lab.id, {}).get("done"))
    print()
    print(ui.bold(f"  {done} of {len(labs)} labs done.") + (f" About {left} minutes of labs left." if left else ""))


def cmd_list(_args) -> int:
    _print_progress(core.find_root())
    return 0


def cmd_start(args) -> int:
    root = core.find_root()
    lab = core.find(root, args.lab)
    readme = lab.folder / "README.md"
    print(readme.read_text(encoding="utf-8"))
    files = ", ".join(str((lab.folder / f).relative_to(root)) for f in lab.files)
    print(ui.coral(f"Open: {files}"))
    print(ui.coral(f"Then run: lab check {lab.id}"))
    return 0


def cmd_hint(args) -> int:
    root = core.find_root()
    lab = core.find(root, args.lab)
    hints = core.hints(lab)
    if not hints:
        print("This lab has no hints.")
        return 0
    progress = core.read_progress(root)
    entry = progress.setdefault(lab.id, {})
    level = min(entry.get("hints", 0) + 1, len(hints))
    entry["hints"] = level
    core.write_progress(root, progress)
    ui.heading(f"{lab.id}  hint {level} of {len(hints)}")
    print(hints[level - 1])
    if level < len(hints):
        print(ui.dim(f"\nStill stuck? Run `lab hint {lab.id}` again for the next one."))
    return 0


def cmd_reset(args) -> int:
    root = core.find_root()
    lab = core.find(root, args.lab)
    if not args.yes:
        answer = input(f"This replaces your work in {lab.slug} with the starting files. Type yes to continue: ")
        if answer.strip().lower() != "yes":
            print("Nothing changed.")
            return 0
    restored = core.reset(lab)
    print(f"Restored {', '.join(restored)} in {lab.slug}.")
    return 0


def cmd_doctor(_args) -> int:
    problems = 0

    def report(ok: bool, name: str, fix: str = "") -> None:
        nonlocal problems
        if ok:
            print(f"  PASS  {name}")
        else:
            problems += 1
            print(f"  {ui.coral('FAIL')}  {name}")
            if fix:
                print(f"        {fix}")

    ui.heading("Checking your setup")
    version = sys.version_info
    report(version >= (3, 11), f"Python {version.major}.{version.minor}", "This repo needs Python 3.11 or newer. Run `uv sync` and uv will fetch it.")
    report(shutil.which("uv") is not None, "uv is installed", "Install uv: https://docs.astral.sh/uv/")
    report(shutil.which("git") is not None, "git is installed", "Install git: https://git-scm.com/downloads")
    try:
        root = core.find_root()
        labs = core.all_labs(root)
        report(bool(labs), f"{len(labs)} labs found")
    except SystemExit as err:
        report(False, "labs folder found", str(err))
    print()
    print(ui.bold("  You are ready. Run: lab list") if not problems else f"  Fix the {problems} item(s) above, then run `lab doctor` again.")
    return 1 if problems else 0


def cmd_verify(args) -> int:
    """For maintainers: every starter must fail its checks and every solution must pass them."""
    root = core.find_root()
    solutions = Path(args.solutions).resolve()
    labs = [core.find(root, args.lab)] if args.lab else core.all_labs(root)
    bad = 0
    ui.heading("Verifying labs")
    for lab in labs:
        notes = []
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / lab.slug
            shutil.copytree(lab.folder, copy, ignore=shutil.ignore_patterns("__pycache__"))
            for source in (lab.folder / core.STARTER_DIR).glob("*"):
                shutil.copyfile(source, copy / source.name)
            starter = core.run_checks(lab, copy)
            if core.passed(starter):
                notes.append("the starter passes every check")
            for name in lab.files:
                if (lab.folder / name).read_bytes() != (lab.folder / core.STARTER_DIR / name).read_bytes():
                    notes.append(f"{name} differs from its saved starting copy")
            answer = solutions / lab.slug
            if not answer.is_dir():
                notes.append("no solution found")
            else:
                for source in answer.iterdir():
                    if source.is_file():
                        shutil.copyfile(source, copy / source.name)
                solved = core.run_checks(lab, copy)
                for result in solved:
                    if not result["ok"]:
                        notes.append(f"solution fails '{result['title']}': {result['message'].splitlines()[0]}")
        failing = sum(not r["ok"] for r in starter)
        if notes:
            bad += 1
            print(f"  {ui.coral('FAIL')}  {lab.id:<8} {'; '.join(notes)}")
        else:
            print(f"  PASS  {lab.id:<8} starter fails {failing} of {len(starter)}, solution passes {len(starter)}")
    print()
    print(ui.bold(f"  {len(labs) - bad} of {len(labs)} labs verified."))
    return 1 if bad else 0


def main() -> None:
    ui.setup_output()
    parser = argparse.ArgumentParser(prog="lab", description="Run and check the course labs.")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("doctor", help="check your machine is ready").set_defaults(run=cmd_doctor)
    commands.add_parser("list", help="show every lab and your progress").set_defaults(run=cmd_list)
    commands.add_parser("progress", help="show your progress").set_defaults(run=cmd_list)

    start = commands.add_parser("start", help="show a lab's task card")
    start.add_argument("lab")
    start.set_defaults(run=cmd_start)

    check = commands.add_parser("check", help="check your work on a lab")
    check.add_argument("lab", nargs="?")
    check.add_argument("--all", action="store_true", help="check every lab")
    check.add_argument("--strict", action="store_true", help="exit with an error unless everything passes")
    check.add_argument("--summary", action="store_true", help="write a progress table (used by GitHub Actions)")
    check.add_argument("--json", action="store_true", help="print the results as JSON")
    check.set_defaults(run=cmd_check)

    hint = commands.add_parser("hint", help="show the next hint for a lab")
    hint.add_argument("lab")
    hint.set_defaults(run=cmd_hint)

    reset = commands.add_parser("reset", help="put a lab's files back to the start")
    reset.add_argument("lab")
    reset.add_argument("--yes", action="store_true", help="do not ask first")
    reset.set_defaults(run=cmd_reset)

    verify = commands.add_parser("verify", help="maintainers: prove starters fail and solutions pass")
    verify.add_argument("--solutions", required=True, help="folder holding one solution folder per lab")
    verify.add_argument("lab", nargs="?")
    verify.set_defaults(run=cmd_verify)

    args = parser.parse_args()
    sys.exit(args.run(args))


if __name__ == "__main__":
    main()
