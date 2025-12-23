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

def _extract_java_code(generated_text: str) -> str:
    """
    Extract Java code from LLM output, looking for ```java code blocks.
    If no code block found, return the original text.
    Args:
        generated_text: Raw LLM output
    Returns:
        Extracted Java code
    """
    pattern = r'```(?:java)?\s*\n(.*?)```'
    matches = re.findall(pattern, generated_text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return generated_text.strip()


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
    constructors: List[str]
    setters: List[str]
    getters: List[str]
    imports: List[str]
    testTemplate: str
    conditionCount: int
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
    constructors: List[str]
    setters: List[str]
    getters: List[str]
    imports: List[str]
    testTemplate: str
    conditionCount: int
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
    path_coverage_details: List[Dict[str, Any]]  # Details for each method in path

    iteration_log: List[Dict[str, Any]]  
    last_prompt: str 

def build_graph(llm: Runnable, cfg: AppConfig) -> Any:
    """
    Returns a compiled LangGraph app that streams step-by-step activity.
    """

    prompt = PromptTemplate(
        input_variables=[
            "entryPoint", "thirdPartyMethod", "path", "methodSources", 
            "constructors", "setters", "getters", "imports",
            "test_package", "test_class_name", "last_run_output",
            "testTemplate", "conditionCount"

        ],
        template=(
            SYSTEM_PROMPT
            + "\n\n"
            + USER_PROMPT_TEMPLATE
            + "\n\nPrevious Maven test output (if any; fix compilation/runtime issues while keeping ALL constraints):\n"
            + "{last_run_output}\n"
        ),
    )

    def decode_code(code_str):
            """Convert escaped newlines and tabs to actual characters"""
            return code_str.replace('\\n', '\n').replace('\\t', '\t')

    def node_generate(state: AgentState) -> AgentState:
        it = int(state.get("iteration", 0)) + 1
        state["iteration"] = it
        state.setdefault("trace", []).append(f"[Generate] iteration={it}")
        method_sources = [decode_code(src) for src in state["methodSources"]]
        constructors = [decode_code(c) for c in state.get("constructors", [])]
        setters = [decode_code(s) for s in state.get("setters", [])]
        getters = [decode_code(g) for g in state.get("getters", [])]
        
        # Build formatted blocks only if non-empty
        method_sources_str = "```java\n" + "\n\n".join(method_sources) + "\n```" if method_sources else ""
        constructors_str = "```java\n" + "\n\n".join(constructors) + "\n```" if constructors else ""
        setters_str = "```java\n" + "\n\n".join(setters) + "\n```" if setters else ""
        getters_str = "```java\n" + "\n\n".join(getters) + "\n```" if getters else ""
        
        rendered = prompt.format(
            entryPoint=state["entryPoint"],
            thirdPartyMethod=state["thirdPartyMethod"],
            path=" -> ".join(state["path"]),
            methodSources=method_sources_str,
            constructors=constructors_str,
            setters=setters_str,
            getters=getters_str,
            imports=", ".join(state.get("imports", [])),
            testTemplate=decode_code(state.get("testTemplate", "")),
            test_package=state["test_package"],
            test_class_name=state["test_class_name"],
            last_run_output=state.get("last_run_output", "") or "",
        )
        state["last_prompt"] = rendered

        java = llm.invoke(rendered)
        if not isinstance(java, str):
            java = str(java)
        java = _extract_java_code(java)
        java = java.replace("\r\n", "\n").strip() + "\n"
        #_validate_generated_java(java, state["test_package"], state["test_class_name"])
        state["java_source"] = java
        state.setdefault("trace", []).append(f"[Generate] produced {len(java)} chars")
        # Log iteration details
        state.setdefault("iteration_log", [])
        state["iteration_log"].append({
            "iteration": it,
            "prompt": rendered,
            "generated_java": java,
            "maven_feedback": None,
            "coverage": None,
            "approved": None,
        })


        
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
        
        # Update iteration log with Maven results
        state["iteration_log"][-1]["maven_feedback"] = formatted_feedback
        state["iteration_log"][-1]["maven_exit_code"] = rr.exit_code
        state["iteration_log"][-1]["maven_success"] = rr.success

        return state

    def node_check_coverage(state: AgentState) -> AgentState:
        """Check if each method in the path is covered by the test."""
        state.setdefault("trace", []).append("[Coverage] checking path coverage (granular)")
        
        path = state["path"]
        target_method = state["thirdPartyMethod"]
        path_coverage_details = []
        
        # Check coverage for each method in the path (excluding the last one which is the target)
        for i in range(len(path) - 1):
            method_to_check = path[i]
            
            # Determine the method_class for this check
            if i == 0:
                # First method in path - use entryPoint's class
                entry_point = state["entryPoint"]
                method_class = entry_point.rsplit('.', 1)[0]
            else:
                # For subsequent methods, use the previous method in path as the class context
                method_class = path[i - 1]
            
            # Call the coverage tool for this method
            coverage_result = get_coverage_result(
                repo_root=Path(state["repo_root"]),
                method_class=method_class,
                target_method=method_to_check
            )
            
            path_coverage_details.append({
                "index": i,
                "method": method_to_check,
                "method_class": method_class,
                "covered": coverage_result.method_covered,
                "error": coverage_result.error,
            })
            
            state.setdefault("trace", []).append(
                f"[Coverage] path[{i}] '{method_to_check}' covered={coverage_result.method_covered}"
            )
        
        # Check the target method (last in path)
        if len(path) >= 2:
            method_class = path[-2]  # Second-to-last method in the path
        else:
            entry_point = state["entryPoint"]
            method_class = entry_point.rsplit('.', 1)[0]
        
        target_coverage_result = get_coverage_result(
            repo_root=Path(state["repo_root"]),
            method_class=method_class,
            target_method=target_method
        )
        
        path_coverage_details.append({
            "index": len(path) - 1,
            "method": target_method,
            "method_class": method_class,
            "covered": target_coverage_result.method_covered,
            "error": target_coverage_result.error,
        })
        
        # Update state with coverage information
        state["target_method_covered"] = target_coverage_result.method_covered
        state["coverage_total_lines"] = target_coverage_result.total_covered_lines
        state["coverage_error"] = target_coverage_result.error
        state["path_coverage_details"] = path_coverage_details
        
        state.setdefault("trace", []).append(
            f"[Coverage] target '{target_method}' covered={target_coverage_result.method_covered}"
        )
        
        # Update iteration log with coverage results
        state["iteration_log"][-1]["coverage"] = {
            "method_covered": target_coverage_result.method_covered,
            "total_covered_lines": target_coverage_result.total_covered_lines,
            "error": target_coverage_result.error,
            "path_coverage_details": path_coverage_details,
        }
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
            # Add granular coverage feedback to help LLM understand what went wrong
            if not state.get("last_run_output"):
                state["last_run_output"] = ""
            
            # Build detailed coverage feedback
            coverage_feedback = "\n\n" + "=" * 80 + "\n"
            coverage_feedback += "COVERAGE CHECK FAILED\n"
            coverage_feedback += "=" * 80 + "\n"
            
            path_details = state.get("path_coverage_details", [])
            path = state["path"]
            
            if path_details:
                # Find where coverage stops
                last_covered_idx = -1
                for detail in path_details:
                    if detail["covered"]:
                        last_covered_idx = detail["index"]
                    else:
                        break
                
                if last_covered_idx == -1:
                    # No methods in the path are covered
                    coverage_feedback += f"None of the methods in the call path were reached by the test.\n"
                    coverage_feedback += f"The entry point '{state['entryPoint']}' itself may not be properly invoked.\n"
                elif last_covered_idx < len(path) - 1:
                    # Some methods covered but not all
                    covered_method = path_details[last_covered_idx]["method"]
                    next_idx = last_covered_idx + 1
                    uncovered_method = path_details[next_idx]["method"]
                    
                    coverage_feedback += f"The test successfully reaches up to method #{last_covered_idx + 1} in the path:\n"
                    coverage_feedback += f"  ✓ '{covered_method}'\n\n"
                    coverage_feedback += f"However, the next method in the path is NOT covered:\n"
                    coverage_feedback += f"  ✗ '{uncovered_method}' (method #{next_idx + 1})\n\n"
                    
                    if next_idx == len(path) - 1:
                        coverage_feedback += f"This uncovered method is the TARGET third-party method.\n"
                    else:
                        coverage_feedback += f"The call chain breaks here. The target method '{state['thirdPartyMethod']}' "
                        coverage_feedback += f"comes later in the path and is also unreachable.\n"
                    
                    coverage_feedback += f"\nPlease modify the test to ensure '{covered_method}' properly calls '{uncovered_method}'.\n"
                else:
                    # All intermediate methods covered but target is not
                    coverage_feedback += f"All intermediate methods in the path are covered, but the final target method\n"
                    coverage_feedback += f"'{state['thirdPartyMethod']}' was NOT reached.\n\n"
                    coverage_feedback += f"Please ensure the test execution path actually invokes this target method.\n"
                
                # Show full path for context
                coverage_feedback += f"\nFull call path ({len(path)} methods):\n"
                for i, detail in enumerate(path_details):
                    status = "✓" if detail["covered"] else "✗"
                    method_name = detail["method"]
                    if i == len(path_details) - 1:
                        coverage_feedback += f"  {status} #{i + 1}. {method_name} (TARGET)\n"
                    else:
                        coverage_feedback += f"  {status} #{i + 1}. {method_name}\n"
            else:
                # Fallback if path details not available
                coverage_feedback += f"The test compiled and ran successfully, but the target method\n"
                coverage_feedback += f"'{state['thirdPartyMethod']}' was NOT reached/covered by the test.\n"
                coverage_feedback += f"Please modify the test to ensure it actually invokes this third-party method\n"
                coverage_feedback += f"through the entry point '{state['entryPoint']}'.\n"
            
            coverage_feedback += "=" * 80 + "\n"
            state["last_run_output"] += coverage_feedback
        else:
            state["approved"] = False
            state.setdefault("trace", []).append(
                f"[Decide] approved=False (tests failed, will retry if iteration<{max_it})"
            )
        
        # Update iteration log with decision details
        state["iteration_log"][-1]["approved"] = state["approved"]
        state["iteration_log"][-1]["decision_reason"] = (
            "passed+covered" if tests_passed and target_covered
            else "passed_not_covered" if tests_passed
            else "tests_failed"
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
        constructors=inp.constructors,
        setters=inp.setters,
        getters=inp.getters,
        imports=inp.imports,
        testTemplate=inp.testTemplate,
        conditionCount=inp.conditionCount,
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
        path_coverage_details=[],
        iteration_log=[],
        last_prompt="",

    )
