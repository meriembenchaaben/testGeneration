from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, END

from junit_agent.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from junit_agent.tools import run_maven_test, write_test_file


def _java_rel_path(test_package: str, class_name: str) -> str:
    pkg_path = test_package.replace(".", "/")
    return f"src/test/java/{pkg_path}/{class_name}.java"


def _validate_generated_java(java_source: str, test_package: str, test_class_name: str) -> None:
    if not re.search(rf"^\s*package\s+{re.escape(test_package)}\s*;\s*$", java_source, re.MULTILINE):
        raise ValueError(f"Missing or wrong package declaration (expected: {test_package}).")
    if f"class {test_class_name}" not in java_source and f"class\t{test_class_name}" not in java_source:
        raise ValueError(f"Missing or wrong class name (expected: {test_class_name}).")
    if java_source.count("@Test") != 1:
        raise ValueError("Generated source must contain exactly one @Test annotation.")
    if "import" not in java_source:
        raise ValueError("Generated source looks incomplete (missing imports).")


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
    fullMethods: List[str]
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
    fullMethods: List[str]
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


def build_graph(llm: Runnable, cfg: AppConfig) -> Any:
    """
    Returns a compiled LangGraph app that streams step-by-step activity.
    """

    prompt = PromptTemplate(
        input_variables=[
            "entryPoint", "thirdPartyMethod", "path", "fullMethods", "test_package", "test_class_name", "last_run_output"
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
            fullMethods="\n\n".join(state["fullMethods"]),
            test_package=state["test_package"],
            test_class_name=state["test_class_name"],
            last_run_output=state.get("last_run_output", "") or "",
        )

        java = llm.invoke(rendered)
        # HuggingFacePipeline typically returns a string; normalize cautiously
        if not isinstance(java, str):
            java = str(java)

        java = java.replace("\r\n", "\n").strip() + "\n"
        _validate_generated_java(java, state["test_package"], state["test_class_name"])

        state["java_source"] = java
        state.setdefault("trace", []).append(f"[Generate] produced {len(java)} chars")
        return state

    def node_write(state: AgentState) -> AgentState:
        state.setdefault("trace", []).append("[Write] writing Java test file")
        rel_path = _java_rel_path(state["test_package"], state["test_class_name"])
        state["test_rel_path"] = rel_path

        abs_path = write_test_file(Path(state["repo_root"]), rel_path, state["java_source"])
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
        # keep output bounded to avoid ballooning context
        combined = rr.combined
        if len(combined) > 12000:
            combined = combined[-12000:]
        state["last_run_output"] = combined

        state.setdefault("trace", []).append(f"[Run] success={rr.success} exit={rr.exit_code}")
        if combined:
            state.setdefault("trace", []).append(f"[Run] output_tail_chars={len(combined)}")
        return state

    def node_decide(state: AgentState) -> AgentState:
        it = int(state.get("iteration", 0))
        ok = bool(state.get("success", False))
        max_it = int(state.get("max_iterations", cfg.max_iterations))

        if ok:
            state["approved"] = True
            state.setdefault("trace", []).append("[Decide] approved=True (tests passed)")
        else:
            state["approved"] = False
            state.setdefault("trace", []).append(f"[Decide] approved=False (will retry if iteration<{max_it})")
        return state

    def route_after_decide(state: AgentState) -> str:
        if state.get("approved", False):
            return "finalize"
        if int(state.get("iteration", 0)) >= int(state.get("max_iterations", cfg.max_iterations)):
            return "finalize"
        return "generate"

    def node_finalize(state: AgentState) -> AgentState:
        state.setdefault("trace", []).append("[Finalize] done")
        return state

    sg = StateGraph(AgentState)

    sg.add_node("generate", node_generate)
    sg.add_node("write", node_write)
    sg.add_node("run", node_run)
    sg.add_node("decide", node_decide)
    sg.add_node("finalize", node_finalize)

    sg.set_entry_point("generate")
    sg.add_edge("generate", "write")
    sg.add_edge("write", "run")
    sg.add_edge("run", "decide")
    sg.add_conditional_edges("decide", route_after_decide, {"generate": "generate", "finalize": "finalize"})
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
        fullMethods=inp.fullMethods,
        test_package=inp.test_package,
        test_class_name=inp.test_class_name,
        iteration=0,
        last_run_output="",
        success=False,
        approved=False,
        trace=[],
    )
