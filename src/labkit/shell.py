"""Helpers for checks on a Linux machine: run a command as the member, and look at files, users and processes.

Used by Linux labs (Engineering Foundations). Checks run as root, so they can see everything;
`sh()` runs a command as the member by default, so permissions apply to it as they do to them.

    from labkit import CheckFail, check
    from labkit.shell import sh, mode, owner

    @check("deploy.sh is executable by its owner only")
    def _(lab):
        if mode(lab.path("deploy.sh")) != "700":
            raise CheckFail(...)
"""

import grp
import os
import pwd
import shlex
import stat
import subprocess
from pathlib import Path

from labkit.checks import CheckFail

MEMBER = os.environ.get("LAB_USER", "learner")


class Ran:
    """The result of a command: exit code, what it printed, and what it printed as errors."""

    def __init__(self, code: int, out: str, err: str):
        self.code, self.out, self.err = code, out, err
        self.lines = [line.rstrip() for line in out.splitlines()]

    @property
    def ok(self) -> bool:
        return self.code == 0

    def __repr__(self) -> str:
        return f"Ran(code={self.code}, out={self.out[:200]!r}, err={self.err[:200]!r})"


def sh(command: str, user: str | None = MEMBER, cwd=None, timeout: int = 30, env: dict | None = None, stdin: str | None = None) -> Ran:
    """Run a bash command, as the member by default (user=None runs it as whoever runs the checks)."""
    full_env = {**os.environ, **(env or {})}
    if user and os.geteuid() == 0 and _exists_user(user):
        home = pwd.getpwnam(user).pw_dir
        full_env.update(HOME=home, USER=user, LOGNAME=user)
        args = ["su", user, "-s", "/bin/bash", "-c", command]
    else:
        args = ["bash", "-c", command]
    try:
        done = subprocess.run(args, cwd=str(cwd) if cwd else None, capture_output=True, text=True,
                              timeout=timeout, env=full_env, input=stdin if stdin is not None else "")
    except subprocess.TimeoutExpired:
        raise CheckFail(f"`{command}` did not finish within {timeout} seconds. Look for something that waits for input or never ends.")
    return Ran(done.returncode, done.stdout, done.stderr)


def _exists_user(name: str) -> bool:
    try:
        pwd.getpwnam(name)
        return True
    except KeyError:
        return False


def _path(path) -> Path:
    return Path(path)


def need_file(path, what: str = "") -> Path:
    p = _path(path)
    if not p.is_file():
        raise CheckFail(f"There should be a file at `{p}`{' ' + what if what else ''}. There is not one yet.")
    return p


def need_dir(path, what: str = "") -> Path:
    p = _path(path)
    if not p.is_dir():
        raise CheckFail(f"There should be a folder at `{p}`{' ' + what if what else ''}. There is not one yet.")
    return p


def read(path) -> str:
    return need_file(path).read_text(encoding="utf-8", errors="replace")


def mode(path) -> str:
    """Permission bits as octal digits, for example '755'. Special bits show as a fourth digit ('2755' is setgid)."""
    p = _path(path)
    if not p.exists() and not p.is_symlink():
        raise CheckFail(f"`{p}` does not exist.")
    return format(stat.S_IMODE(p.lstat().st_mode), "03o")


def owner(path) -> str:
    return pwd.getpwuid(_path(path).lstat().st_uid).pw_name


def group(path) -> str:
    return grp.getgrgid(_path(path).lstat().st_gid).gr_name


def user_exists(name: str) -> bool:
    return _exists_user(name)


def groups_of(name: str) -> list:
    """Every group a user belongs to, primary group first."""
    try:
        entry = pwd.getpwnam(name)
    except KeyError:
        return []
    primary = grp.getgrgid(entry.pw_gid).gr_name
    return [primary] + sorted(g.gr_name for g in grp.getgrall() if name in g.gr_mem and g.gr_name != primary)


def processes() -> list:
    """Running processes as (pid, user, command line)."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            uid = (entry / "status").read_text().split("Uid:")[1].split()[0]
            found.append((int(entry.name), pwd.getpwuid(int(uid)).pw_name, cmd))
        except (OSError, IndexError, KeyError):
            continue
    return found


def running(fragment: str) -> bool:
    """Whether a process whose command line contains this text is running."""
    return any(fragment in cmd for _, _, cmd in processes())


def git(args: str, cwd, user: str | None = MEMBER) -> Ran:
    """Run git in a folder as the member, with output that does not depend on their settings."""
    return sh("git -c color.ui=never -c core.pager=cat " + args, user=user, cwd=cwd)


def quote(text: str) -> str:
    return shlex.quote(text)
