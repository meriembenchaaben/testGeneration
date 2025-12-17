from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RunResult:
    success: bool
    exit_code: int
    stdout: str
    stderr: str

    @property
    def combined(self) -> str:
        out = (self.stdout or "").strip()
        err = (self.stderr or "").strip()
        if out and err:
            return f"{out}\n{err}"
        return out or err


def write_test_file(repo_root: Path, rel_path: str, content: str) -> Path:
    """
    Tool: write the generated Java test under the Maven repo.
    Constrained: always under repo_root, always a text write.
    todo: Yogya will provide the endpoint for this. (the tool)
    """
    abs_path = (repo_root / rel_path).resolve()
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(content, encoding="utf-8")
    return abs_path


def run_maven_test(repo_root: Path, mvn_cmd: str, test_fqcn: Optional[str], timeout_s: int = 300) -> RunResult:
    """
    Tool: run Maven tests. If test_fqcn is given, runs only that test via -Dtest=<ClassName>.
    Constrained: only runs mvn test, no arbitrary shell.
    todo: Yogya will provide the endpoint for this. (the tool)
    Ideally, we want these two functions seperated.
    # To be changed later.

    """
    cmd = [mvn_cmd, "-q", "test"]
    if test_fqcn:
        simple = test_fqcn.split(".")[-1]
        cmd = [mvn_cmd, "-q", f"-Dtest={simple}", "test"]

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return RunResult(
        success=(proc.returncode == 0),
        exit_code=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )
