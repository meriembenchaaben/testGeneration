from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from junit_agent.write_test_file_ import write_test_file as _write_test_impl, get_test_class_fqn, cleanup_test_file
from junit_agent.run_maven_test_ import run_maven_test as _run_maven_impl, RunResult as RunResultImpl
from junit_agent.get_coverage_result_ import  get_coverage_result as _get_coverage_impl,  CoverageResult as _CoverageResultImpl

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


@dataclass(frozen=True)
class CoverageResult:
    method_covered: bool
    method_class: str
    target_method: str
    total_covered_lines: int
    error: Optional[str] = None


def write_test(repo_root: Path, rel_path: str, content: str) -> Path:
    """
    Tool: write the generated Java test under the Maven repo.
    Args:
        repo_root: Root directory of the Maven project
        rel_path: Relative path hint / filename
        content: Complete Java test file content as string
    Returns:
        Path: Absolute path to the written test file
    """
    try:
        return _write_test_impl(repo_root, rel_path, content)
    except Exception as e:
        logger.error(f"Failed to write test file: {e}")
        abs_path = (repo_root / rel_path).resolve()
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")
        logger.warning(f"Used fallback write method for: {abs_path}")
        return abs_path


def run_maven_test(
    repo_root: Path,
    mvn_cmd: str = "mvn",
    test_fqcn: Optional[str] = None,
    timeout_s: int = 300
) -> RunResult:
    """
    Tool: run Maven tests with JaCoCo coverage generation.
    Args:
        repo_root: Root directory of the Maven project
        mvn_cmd: Maven command to use
        test_fqcn: Fully qualified test class name 
                   If None, runs all tests
        timeout_s: Timeout in seconds
    Returns:
        RunResult: Object containing success status, exit code, stdout, and stderr
    """
    try:
        impl_result = _run_maven_impl(
            repo_root=repo_root,
            mvn_cmd="mvn",
            test_fqcn=test_fqcn,
            timeout_s=timeout_s,
            with_jacoco=True,
            skip_checkstyle=True
        )
        return RunResult(
            success=impl_result.success,
            exit_code=impl_result.exit_code,
            stdout=impl_result.stdout,
            stderr=impl_result.stderr
        )
    except Exception as e:
        logger.error(f"Failed to run Maven test: {e}")
        return RunResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr=f"Exception during test execution: {str(e)}"
        )


def get_coverage_result(
    repo_root: Path,
    method_class: str,
    target_method: str
) -> CoverageResult:
    """
    Tool: analyze JaCoCo coverage report to check if a third-party method is covered.
    Args:
        repo_root: Root directory of the Maven project
        method_class: Fully qualified class name being tested
        target_method: Fully qualified third-party method to check 
    Returns:write_test_file
        CoverageResult: Object containing coverage status and details
    """
    try:
        impl_result = _get_coverage_impl(
            repo_root=repo_root,
            method_class=method_class,
            target_method=target_method
        )
        return CoverageResult(
            method_covered=impl_result.method_covered,
            method_class=impl_result.method_class,
            target_method=impl_result.target_method,
            total_covered_lines=impl_result.total_covered_lines,
            error=impl_result.error
        )
    except Exception as e:
        logger.error(f"Failed to get coverage result: {e}")
        return CoverageResult(
            method_covered=False,
            method_class=method_class,
            target_method=target_method,
            total_covered_lines=0,
            error=f"Exception during coverage analysis: {str(e)}"
        )

def extract_test_class_name(test_content: str, filename: Optional[str] = None) -> str:
    """
    Helper: extract fully qualified test class name from test file content.
    Args:
        test_content: Java test file content
        filename: Optional filename to use as fallback
    Returns:
        Fully qualified class name
    """
    try:
        return get_test_class_fqn(test_content, fallback_filename=filename)
    except Exception as e:
        logger.error(f"Failed to extract test class name: {e}")
        if filename:
            return filename.replace('.java', '')
        raise


def cleanup_test(test_file_path: Path) -> None:
    """
    Helper: remove a test file and clean up empty parent directories.
    Args:
        test_file_path: Path to the test file to remove
    """
    try:
        cleanup_test_file(test_file_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup test file: {e}")
