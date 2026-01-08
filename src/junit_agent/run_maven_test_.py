#!/usr/bin/env python3
"""
Module for running Maven tests with optional coverage.

This module provides functionality to:
1. Clean Maven build artifacts
2. Run specific test classes or all tests
3. Generate JaCoCo coverage reports
4. Capture and structure test execution results
"""

import subprocess
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

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


def run_maven_clean(repo_root: Path) -> RunResult:
    """
    Run Maven clean to remove previous build artifacts.
    Also explicitly deletes the target directory including coverage reports
    to ensure a clean state.
    Args:
        repo_root: Root directory of the Maven project
    Returns:
        RunResult with execution details
    """
    # First, manually delete the target directory to ensure coverage reports are removed
    target_dir = repo_root / "target"
    if target_dir.exists():
        logger.info(f"Manually deleting target directory: {target_dir}")
        try:
            shutil.rmtree(target_dir)
            logger.info("Target directory deleted successfully")
        except Exception as e:
            logger.warning(f"Failed to manually delete target directory: {e}")
    
    clean_cmd = ["mvn", "clean", "-f", str(repo_root / "pom.xml")]
    logger.info("Running Maven clean...")
    original_dir = os.getcwd()
    try:
        os.chdir(repo_root)
        proc = subprocess.run(
            clean_cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        if proc.returncode == 0:
            logger.info("Maven clean completed successfully")
        else:
            logger.warning(f"Maven clean failed with exit code {proc.returncode}")
        return RunResult(
            success=(proc.returncode == 0),
            exit_code=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )
    except subprocess.TimeoutExpired:
        logger.error("Maven clean timed out")
        return RunResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr="Maven clean timed out after 2 minutes",  # here, maybe we can give it more time, if a timeout occurs
        )
    except Exception as e:
        logger.error(f"Maven clean failed with exception: {e}")
        return RunResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr=str(e),
        )
    finally:
        os.chdir(original_dir)

def run_maven_test(
    repo_root: Path,
    mvn_cmd: str = "mvn",
    test_fqcn: Optional[str] = None,
    timeout_s: int = 300,
    with_jacoco: bool = True,
    skip_checkstyle: bool = True
) -> RunResult:
    """
    Run Maven tests with optional JaCoCo coverage.
    Args:
        repo_root: Root directory of the Maven project
        mvn_cmd: Maven command to run (default: "mvn")
        test_fqcn: Fully qualified class name of specific test to run
                   If None, runs all tests
        timeout_s: Timeout in seconds (default: 300)
        with_jacoco: Whether to generate JaCoCo coverage report (default: True)
        skip_checkstyle: Whether to skip checkstyle checks (default: True)
    Returns:
        RunResult with execution details including stdout, stderr, and exit code
    Raises:
        ValueError: If repo_root doesn't contain a valid pom.xml
    """
    pom_file = repo_root / "pom.xml"
    if not pom_file.exists():
        raise ValueError(f"No pom.xml found in {repo_root}")
    cmd = [mvn_cmd, "clean", "test"]
    if test_fqcn:
        cmd.append(f"-Dtest={test_fqcn}")
        logger.info(f"Running specific test class: {test_fqcn}")
    else:
        logger.info("Running all tests")
    if skip_checkstyle:
        cmd.append("-Dcheckstyle.skip=true")
    cmd.extend(["-f", str(pom_file)])
    logger.info(f"Executing: {' '.join(cmd)}")
    logger.info(f"Working directory: {repo_root}")
    logger.info(f"Timeout: {timeout_s} seconds")
    original_dir = os.getcwd()
    try:
        os.chdir(repo_root)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            shell=False
        )
        success = (proc.returncode == 0)
        if success:
            logger.info("Maven test execution completed successfully")
        else:
            logger.warning(f"Maven test execution failed with exit code {proc.returncode}")
        if proc.stdout:
            stdout_lines = proc.stdout.count('\n')
            logger.debug(f"Captured {stdout_lines} lines of stdout")
        if proc.stderr:
            stderr_lines = proc.stderr.count('\n')
            logger.debug(f"Captured {stderr_lines} lines of stderr")
        return RunResult(
            success=success,
            exit_code=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )
    except subprocess.TimeoutExpired as e:
        logger.error(f"Maven test timed out after {timeout_s} seconds")
        stdout = e.stdout.decode('utf-8') if e.stdout else ""
        stderr = e.stderr.decode('utf-8') if e.stderr else ""
        return RunResult(
            success=False,
            exit_code=-1,
            stdout=stdout,
            stderr=f"Timeout after {timeout_s} seconds\n{stderr}",
        )
    except Exception as e:
        logger.error(f"Maven test execution failed with exception: {e}", exc_info=True)
        return RunResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr=f"Exception during test execution: {str(e)}",
        )
    finally:
        os.chdir(original_dir)

def run_maven_test_with_clean(
    repo_root: Path,
    test_fqcn: Optional[str] = None,
    timeout_s: int = 300
) -> RunResult:
    """
    Run Maven clean followed by test execution with JaCoCo coverage.
    This is a convenience function that combines clean and test operations.
    Args:
        repo_root: Root directory of the Maven project
        test_fqcn: Fully qualified class name of specific test to run
        timeout_s: Timeout in seconds for the test execution (not including clean)
    Returns:
        RunResult from the test execution (clean result is logged but not returned)
    """
    clean_result = run_maven_clean(repo_root)
    if not clean_result.success:
        logger.warning("Maven clean failed, but continuing with test execution")
    return run_maven_test(
        repo_root=repo_root,
        test_fqcn=test_fqcn,
        timeout_s=timeout_s,
        with_jacoco=True,
        skip_checkstyle=True
    )
