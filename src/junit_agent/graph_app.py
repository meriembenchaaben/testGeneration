from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, END

from junit_agent.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from junit_agent.tools import run_maven_test, write_test, get_coverage_result


def _java_rel_path(test_package: str, class_name: str) -> str:
    pkg_path = test_package.replace(".", "/")
    return f"src/test/java/{pkg_path}/{class_name}.java"


def _validate_generated_java(java_source: str, test_package: str, test_class_name: str) -> None:
    if not re.search(rf"^\s*package\s+{re.escape(test_package)}\s*;\s*$", java_source, re.MULTILINE):
        raise ValueError(f"Missing or wrong package declaration (expected: {test_package}).")
    if f"class {test_class_name}" not in java_source and f"class\t{test_class_name}" not in java_source:
        raise ValueError(f"Missing or wrong class name (expected: {test_class_name}).")
    if java_source.count("@Test") < 1:
        raise ValueError("Generated source must contain at least one @Test annotation.")
    if "import" not in java_source:
        raise ValueError("Generated source looks incomplete (missing imports).")


def _extract_maven_errors(output: str) -> str:
    """
    Extract all lines containing [ERROR] from Maven output.
    Also includes context lines (lines immediately before/after errors) for better understanding.
    
    Args:
        output: Combined Maven stdout/stderr output
    
    Returns:
        Formatted error report with Maven errors and context
    """
    if not output:
        return ""
    
    lines = output.split('\n')
    error_lines = []
    error_indices = set()
    
    # First pass: find all lines with [ERROR]
    for i, line in enumerate(lines):
        if '[ERROR]' in line:
            error_indices.add(i)
    
    if not error_indices:
        return ""
    
    # Second pass: collect errors with context (1 line before and after)
    collected_indices = set()
    for idx in error_indices:
        # Add the error line and surrounding context
        for context_idx in range(max(0, idx - 1), min(len(lines), idx + 2)):
            collected_indices.add(context_idx)
    
    # Build the error report
    error_report = ["=" * 80, "MAVEN ERROR REPORT", "=" * 80, ""]
    
    # Collect lines in order
    sorted_indices = sorted(collected_indices)
    for i, idx in enumerate(sorted_indices):
        line = lines[idx]
        
        # Add separator for non-consecutive lines
        if i > 0 and idx - sorted_indices[i-1] > 1:
            error_report.append("...")
            error_report.append("")
        
        # Mark error lines clearly
        if '[ERROR]' in line:
            error_report.append(f">>> {line}")
        else:
            error_report.append(f"    {line}")
    
    error_report.extend(["", "=" * 80, ""])
    
    return '\n'.join(error_report)


def _extract_compilation_errors(output: str) -> str:
    """
    Extract compilation errors with more detail.
    Looks for common compilation error patterns.
    
    Args:
        output: Maven output
    
    Returns:
        Formatted compilation error details
    """
    if "COMPILATION ERROR" not in output and "compilation failure" not in output.lower():
        return ""
    
    lines = output.split('\n')
    comp_errors = []
    in_error_section = False
    
    for line in lines:
        # Start of compilation error section
        if "COMPILATION ERROR" in line or "[ERROR] COMPILER ERROR" in line:
            in_error_section = True
            comp_errors.append("=" * 80)
            comp_errors.append("COMPILATION ERRORS DETECTED")
            comp_errors.append("=" * 80)
            comp_errors.append("")
            continue
        
        # Collect error details
        if in_error_section:
            # Stop at BUILD FAILURE or next major section
            if "BUILD FAILURE" in line or "BUILD SUCCESS" in line:
                in_error_section = False
                comp_errors.append("")
                continue
            
            # Include lines that look like error messages
            if any(marker in line for marker in [
                "[ERROR]", "error:", "symbol:", "location:", "cannot find", 
                "package", "does not exist", "incompatible types"
            ]):
                comp_errors.append(line)
    
    if comp_errors:
        comp_errors.append("=" * 80)
        comp_errors.append("")
        return '\n'.join(comp_errors)
    
    return ""


def _extract_test_failures(output: str) -> str:
    """
    Extract test failure details including assertion errors and stack traces.
    
    Args:
        output: Maven output
    
    Returns:
        Formatted test failure details
    """
    if "Tests run:" not in output:
        return ""
    
    lines = output.split('\n')
    failures = []
    in_failure_section = False
    
    for i, line in enumerate(lines):
        # Look for test failure indicators
        if any(marker in line for marker in ["FAILURE!", "ERROR!", "AssertionError", "Failed tests:"]):
            if not in_failure_section:
                failures.append("=" * 80)
                failures.append("TEST FAILURES DETECTED")
                failures.append("=" * 80)
                failures.append("")
                in_failure_section = True
            
            # Include the failure line and next few lines for stack trace
            failures.append(line)
            
            # Grab stack trace (next 5 lines or until blank line)
            for j in range(i + 1, min(i + 6, len(lines))):
                next_line = lines[j]
                if next_line.strip():
                    failures.append(next_line)
                else:
                    break
    
    if failures:
        failures.append("")
        failures.append("=" * 80)
        failures.append("")
        return '\n'.join(failures)
    
    return ""


def _format_feedback(output: str) -> str:
    """
    Format Maven output into structured feedback for the LLM.
    Extracts and highlights errors, compilation issues, and test failures.
    
    Args:
        output: Raw Maven output
    
    Returns:
        Formatted feedback string with key errors highlighted
    """
    if not output:
        return ""
    
    feedback_parts = []
    
    # Extract compilation errors
    comp_errors = _extract_compilation_errors(output)
    if comp_errors:
        feedback_parts.append(comp_errors)
    
    # Extract test failures
    test_failures = _extract_test_failures(output)
    if test_failures:
        feedback_parts.append(test_failures)
    
    # Extract general Maven errors
    maven_errors = _extract_maven_errors(output)
    if maven_errors and not comp_errors and not test_failures:
        # Only include if we haven't already captured errors above
        feedback_parts.append(maven_errors)
    
    # If we found structured errors, return those
    if feedback_parts:
        return '\n'.join(feedback_parts)
    
    # Fallback: return last portion of output if no structured errors found
    if len(output) > 12000:
        return "=" * 80 + "\nMAVEN OUTPUT (last 12000 chars):\n" + "=" * 80 + "\n" + output[-12000:]
    
    return output


@dataclass(frozen=True)
class AppConfig:
    repo_root: Path
    mvn_cmd: str = "mvn"
    max_iterations: int = 5
    run_only_generated_test: bool = True


@dataclass(frozen=True)
class GenerationInput:
    entryPoint: str
    thirdPartyMethod: str
    path: List[str]
    methodSources: List[str]
    test_package: str = "generated"
    test_class_name: str = "GeneratedReachabilityTest"


class AgentState(TypedDict, total=False):
    # immutable config-ish
    repo_root: str
    mvn_cmd: str
    max_iterations: int
    run_only_generated_test: bool

    # generation input
    entryPoint: str
    thirdPartyMethod: str
    path: List[str]
    methodSources: List[str]
    test_package: str
    test_class_name: str

    # loop state
    iteration: int
    java_source: str
    last_run_output: str
    last_exit_code: int
    success: bool
    approved: bool
    test_rel_path: str
    trace: List[str]
    
    # coverage state
    target_method_covered: bool
    coverage_total_lines: int
    coverage_error: Optional[str]


def build_graph(llm: Runnable, cfg: AppConfig) -> Any:
    """
    Returns a compiled LangGraph app that streams step-by-step activity.
    """

    prompt = PromptTemplate(
        input_variables=[
            "entryPoint", "thirdPartyMethod", "path", "methodSources", 
            "test_package", "test_class_name", "last_run_output"
        ],
        template=(
            SYSTEM_PROMPT
            + "\n\n"
            + USER_PROMPT_TEMPLATE
            + "\n\nPrevious Maven test output (if any; fix compilation/runtime issues while keeping ALL constraints):\n"
            + "{last_run_output}\n"
        ),
    )

    def node_generate(state: AgentState) -> AgentState:
        it = int(state.get("iteration", 0)) + 1
        state["iteration"] = it
        state.setdefault("trace", []).append(f"[Generate] iteration={it}")

        rendered = prompt.format(
            entryPoint=state["entryPoint"],
            thirdPartyMethod=state["thirdPartyMethod"],
            path=" -> ".join(state["path"]),
            methodSources="\n\n".join(state["methodSources"]),
            test_package=state["test_package"],
            test_class_name=state["test_class_name"],
            last_run_output=state.get("last_run_output", "") or "",
        )

        java = llm.invoke(rendered)
        # HuggingFacePipeline typically returns a string; normalize cautiously
        if not isinstance(java, str):
            java = str(java)

        java = java.replace("\r\n", "\n").strip() + "\n"
        #_validate_generated_java(java, state["test_package"], state["test_class_name"])

        state["java_source"] = java
        state.setdefault("trace", []).append(f"[Generate] produced {len(java)} chars")
        return state

    def node_write(state: AgentState) -> AgentState:
        state.setdefault("trace", []).append("[Write] writing Java test file")
        rel_path = _java_rel_path(state["test_package"], state["test_class_name"])
        state["test_rel_path"] = rel_path

        abs_path = write_test(Path(state["repo_root"]), rel_path, state["java_source"])
        state.setdefault("trace", []).append(f"[Write] wrote to {abs_path}")
        return state

    def node_run(state: AgentState) -> AgentState:
        state.setdefault("trace", []).append("[Run] running Maven tests")
        test_fqcn = f"{state['test_package']}.{state['test_class_name']}"
        rr = run_maven_test(
            repo_root=Path(state["repo_root"]),
            mvn_cmd=state["mvn_cmd"],
            test_fqcn=test_fqcn if state.get("run_only_generated_test", True) else None,
            timeout_s=300,
        )
        state["success"] = rr.success
        state["last_exit_code"] = rr.exit_code
        
        # Format the output with extracted errors for better LLM feedback
        combined = rr.combined
        formatted_feedback = _format_feedback(combined)
        state["last_run_output"] = formatted_feedback

        state.setdefault("trace", []).append(f"[Run] success={rr.success} exit={rr.exit_code}")
        if formatted_feedback:
            state.setdefault("trace", []).append(f"[Run] feedback_chars={len(formatted_feedback)}")
        
        return state

    def node_check_coverage(state: AgentState) -> AgentState:
        """Check if the target third-party method is covered by the test."""
        state.setdefault("trace", []).append("[Coverage] checking target method coverage")
        
        # Extract method_class from entryPoint (e.g., "tech.tablesaw.io.jsonl.JsonlReader.read")
        entry_point = state["entryPoint"]
        method_class = entry_point.rsplit('.', 1)[0]  # Remove method name to get class
        
        target_method = state["thirdPartyMethod"]
        
        # Call the coverage tool
        coverage_result = get_coverage_result(
            repo_root=Path(state["repo_root"]),
            method_class=method_class,
            target_method=target_method
        )
        
        # Update state with coverage information
        state["target_method_covered"] = coverage_result.method_covered
        state["coverage_total_lines"] = coverage_result.total_covered_lines
        state["coverage_error"] = coverage_result.error
        
        if coverage_result.error:
            state.setdefault("trace", []).append(
                f"[Coverage] error: {coverage_result.error}"
            )
        else:
            state.setdefault("trace", []).append(
                f"[Coverage] target_covered={coverage_result.method_covered}, "
                f"total_lines={coverage_result.total_covered_lines}"
            )
        
        return state

    def node_decide(state: AgentState) -> AgentState:
        it = int(state.get("iteration", 0))
        tests_passed = bool(state.get("success", False))
        target_covered = bool(state.get("target_method_covered", False))
        max_it = int(state.get("max_iterations", cfg.max_iterations))

        # Test is approved only if both conditions are met:
        # 1. Maven tests passed (no compilation/runtime errors)
        # 2. Target third-party method is covered
        if tests_passed and target_covered:
            state["approved"] = True
            state.setdefault("trace", []).append(
                "[Decide] approved=True (tests passed AND target method covered)"
            )
        elif tests_passed and not target_covered:
            state["approved"] = False
            state.setdefault("trace", []).append(
                f"[Decide] approved=False (tests passed but target method NOT covered, "
                f"will retry if iteration<{max_it})"
            )
            # Add coverage feedback to help LLM understand what went wrong
            if not state.get("last_run_output"):
                state["last_run_output"] = ""
            state["last_run_output"] += (
                "\n\n" + "=" * 80 + "\n"
                "COVERAGE CHECK FAILED\n"
                "=" * 80 + "\n"
                f"The test compiled and ran successfully, but the target method "
                f"'{state['thirdPartyMethod']}' was NOT reached/covered by the test.\n"
                f"Please modify the test to ensure it actually invokes this third-party method "
                f"through the entry point '{state['entryPoint']}'.\n"
                "=" * 80 + "\n"
            )
        else:
            state["approved"] = False
            state.setdefault("trace", []).append(
                f"[Decide] approved=False (tests failed, will retry if iteration<{max_it})"
            )
        
        return state

    def route_after_decide(state: AgentState) -> str:
        if state.get("approved", False):
            return "finalize"
        if int(state.get("iteration", 0)) >= int(state.get("max_iterations", cfg.max_iterations)):
            return "finalize"
        return "generate"

    def node_finalize(state: AgentState) -> AgentState:
        state.setdefault("trace", []).append("[Finalize] done")
        
        # Add final summary to trace
        if state.get("approved", False):
            state.setdefault("trace", []).append(
                "[Finalize] SUCCESS: Test passed and target method covered"
            )
        else:
            reason = "unknown"
            if not state.get("success", False):
                reason = "tests failed"
            elif not state.get("target_method_covered", False):
                reason = "target method not covered"
            
            state.setdefault("trace", []).append(
                f"[Finalize] FAILURE: {reason} after {state.get('iteration', 0)} iterations"
            )
        
        return state

    sg = StateGraph(AgentState)

    # Add all nodes
    sg.add_node("generate", node_generate)
    sg.add_node("write", node_write)
    sg.add_node("run", node_run)
    sg.add_node("check_coverage", node_check_coverage)  # New node
    sg.add_node("decide", node_decide)
    sg.add_node("finalize", node_finalize)

    # Define the flow
    sg.set_entry_point("generate")
    sg.add_edge("generate", "write")
    sg.add_edge("write", "run")
    sg.add_edge("run", "check_coverage")  # Check coverage after running tests
    sg.add_edge("check_coverage", "decide")  # Then decide based on coverage
    sg.add_conditional_edges(
        "decide", 
        route_after_decide, 
        {"generate": "generate", "finalize": "finalize"}
    )
    sg.add_edge("finalize", END)

    app = sg.compile()
    return app


def initial_state(inp: GenerationInput, cfg: AppConfig) -> AgentState:
    return AgentState(
        repo_root=str(cfg.repo_root.resolve()),
        mvn_cmd=cfg.mvn_cmd,
        max_iterations=cfg.max_iterations,
        run_only_generated_test=cfg.run_only_generated_test,
        entryPoint=inp.entryPoint,
        thirdPartyMethod=inp.thirdPartyMethod,
        path=inp.path,
        methodSources=inp.methodSources,
        test_package=inp.test_package,
        test_class_name=inp.test_class_name,
        iteration=0,
        last_run_output="",
        success=False,
        approved=False,
        trace=[],
        # Initialize coverage state
        target_method_covered=False,
        coverage_total_lines=0,
        coverage_error=None,
    )
