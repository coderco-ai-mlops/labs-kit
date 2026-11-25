"""Finding labs, running their checks, and remembering progress."""

import json
import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from labkit.worker import MARK

TIMEOUT_SECONDS = 10
STARTER_DIR = ".starter"


@dataclass
class LabInfo:
    folder: Path
    id: str
    title: str
    minutes: int
    section: str
    files: list = field(default_factory=lambda: ["exercise.py"])
    checks: str = "checks.py"
    where: str = "browser"      # "browser": also runs on the labs site. "terminal": needs a real environment.
    packages: list = field(default_factory=list)   # third-party packages the lab imports
    standins: list = field(default_factory=list)   # tools the labs site replaces with a practice version
    group: str = ""             # the dependency group that installs them: uv sync --group <group>
    timeout: int = TIMEOUT_SECONDS

    @property
    def slug(self) -> str:
        return self.folder.name


def find_root(start: Path | None = None) -> Path:
    """The lab repo is the nearest folder, going up, that has a labs/ folder."""
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / "labs").is_dir():
            return folder
    raise SystemExit("No labs/ folder found. Run this from inside a lab repo.")


def load(folder: Path) -> LabInfo:
    data = tomllib.loads((folder / "lab.toml").read_text(encoding="utf-8"))
    return LabInfo(
        folder=folder,
        id=data["id"],
        title=data["title"],
        minutes=int(data.get("minutes", 0)),
        section=data.get("section", ""),
        files=list(data.get("files", ["exercise.py"])),
        checks=data.get("checks", "checks.py"),
        where=data.get("where", "browser"),
        packages=list(data.get("packages", [])),
        standins=list(data.get("standins", [])),
        group=data.get("group", ""),
        timeout=int(data.get("timeout", TIMEOUT_SECONDS)),
    )


def all_labs(root: Path) -> list:
    return [load(p) for p in sorted((root / "labs").iterdir()) if (p / "lab.toml").exists()]


def find(root: Path, wanted: str) -> LabInfo:
    wanted = wanted.lower().strip()
    labs = all_labs(root)
    exact = [lab for lab in labs if wanted in (lab.id.lower(), lab.slug.lower())]
    matches = exact or [lab for lab in labs if lab.slug.lower().startswith(wanted)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(f"No lab called '{wanted}'. Run `lab list` to see them all.")
    raise SystemExit(f"'{wanted}' matches more than one lab: {', '.join(m.id for m in matches)}")


def run_checks(lab: LabInfo, folder: Path | None = None) -> list:
    """Run a lab's checks in a separate process. `folder` lets maintainers check a copy."""
    folder = folder or lab.folder
    command = [sys.executable, "-m", "labkit.worker", str(folder), lab.checks, lab.files[0]]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=lab.timeout, cwd=folder, env=env)
    except subprocess.TimeoutExpired:
        return [{"title": "Your code finishes", "ok": False, "hint": "",
                 "message": f"Your code did not finish in {lab.timeout} seconds. Look for a loop that never ends."}]
    marker = done.stdout.rfind(MARK)
    if marker == -1 and "ModuleNotFoundError" in done.stderr and lab.group:
        return [{"title": "The lab's packages are installed", "ok": False, "hint": "",
                 "message": f"This lab needs extra packages. Install them with: uv sync --group {lab.group}"}]
    if marker == -1:
        detail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["no output"]
        return [{"title": "Checks run", "ok": False, "hint": "",
                 "message": f"The checks could not run: {detail[0]}"}]
    return json.loads(done.stdout[marker + len(MARK):].strip())


def passed(results: list) -> bool:
    return bool(results) and all(r["ok"] for r in results)


# Progress is kept in the repo so it follows the member between machines and Codespaces.

def _progress_file(root: Path) -> Path:
    return root / ".lab" / "progress.json"


def read_progress(root: Path) -> dict:
    try:
        return json.loads(_progress_file(root).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_progress(root: Path, progress: dict) -> None:
    path = _progress_file(root)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record(root: Path, lab: LabInfo, results: list) -> None:
    progress = read_progress(root)
    entry = progress.setdefault(lab.id, {})
    entry["passed"] = sum(r["ok"] for r in results)
    entry["total"] = len(results)
    entry["done"] = entry.get("done", False) or passed(results)
    write_progress(root, progress)


def reset(lab: LabInfo) -> list:
    """Put the lab's files back to how they started. Returns the files restored."""
    starter = lab.folder / STARTER_DIR
    if not starter.is_dir():
        raise SystemExit(f"{lab.id} has no saved starting files to restore.")
    restored = []
    for source in sorted(starter.iterdir()):
        shutil.copyfile(source, lab.folder / source.name)
        restored.append(source.name)
    return restored


def hints(lab: LabInfo) -> list:
    """hints.md holds up to three hints, each under its own '## ' heading."""
    path = lab.folder / "hints.md"
    if not path.exists():
        return []
    parts = path.read_text(encoding="utf-8").split("\n## ")
    return [part.split("\n", 1)[1].strip() for part in parts[1:] if "\n" in part]
