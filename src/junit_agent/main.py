from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from junit_agent.hf_model import build_local_hf_llm
from junit_agent.graph_app import AppConfig, GenerationInput, build_graph, initial_state


def load_input(json_path: Path) -> GenerationInput:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return GenerationInput(
        entryPoint=data["entryPoint"],
        thirdPartyMethod=data["thirdPartyMethod"],
        path=list(data["path"]),
        methodSources=list(data["methodSources"]),
        test_package=data.get("testPackage", "generated"),
        test_class_name=data.get("testClassName", "GeneratedReachabilityTest"),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="langgraph-junit-agent")
    p.add_argument("repo_root", type=str, help="Path to the target Maven Java project")
    p.add_argument("input_json", type=str, help="Path to JSON input containing entryPoint/path/methodSources")
    p.add_argument("--mvn", type=str, default="mvn", help="Maven command (default: mvn)")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face model id")
    p.add_argument("--iters", type=int, default=5, help="Max generate-run iterations")
    p.add_argument("--temp", type=float, default=0.1, help="Sampling temperature")
    p.add_argument("--max-new", type=int, default=2048, help="Max new tokens")
    p.add_argument("--run-all", action="store_true", help="Run full mvn test suite (default runs only generated test)")
    p.add_argument("--output-json", type=str, help="Optional: Save final state to JSON file")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    inp = load_input(Path(args.input_json).resolve())

    cfg = AppConfig(
        repo_root=repo_root,
        mvn_cmd=args.mvn,
        max_iterations=args.iters,
        run_only_generated_test=(not args.run_all),
    )

    llm = build_local_hf_llm(
        model_id=args.model,
        temperature=args.temp,
        max_new_tokens=args.max_new,
    )

    app = build_graph(llm=llm, cfg=cfg)
    state = initial_state(inp, cfg)

    final_state = None
    for event in app.stream(state):
        for node, st in event.items():
            trace = st.get("trace", [])
            if trace:
                print(trace[-1])
            final_state = st

    if not final_state:
        print("No final state produced.")
        return 2

    # Extract key results
    approved = bool(final_state.get("approved", False))
    test_rel_path = final_state.get("test_rel_path", "")
    java_source = final_state.get("java_source", "")
    last_output = final_state.get("last_run_output", "")
    iteration = final_state.get("iteration", 0)
    success = final_state.get("success", False)
    
    # Extract coverage results
    target_covered = final_state.get("target_method_covered", False)
    coverage_lines = final_state.get("coverage_total_lines", 0)
    coverage_error = final_state.get("coverage_error")

    # Print summary
    # print("\n" + "=" * 80)
    # print("=== FINAL RESULT ===")
    # print("=" * 80)
    # print(f"Approved: {approved}")
    # print(f"Iterations: {iteration}/{cfg.max_iterations}")
    # print(f"Tests Passed: {success}")
    # print(f"Target Method Covered: {target_covered}")
    # if coverage_lines > 0:
    #     print(f"Total Covered Lines: {coverage_lines}")
    # if coverage_error:
    #     print(f"Coverage Error: {coverage_error}")
    # print(f"Test File: {test_rel_path}")
    # print("=" * 80)

    # Print failure reason if not approved
    # if not approved:
    #     print("\n=== FAILURE REASON ===")
    #     if not success:
    #         print("Tests failed to compile or run successfully")
    #     elif not target_covered:
    #         print("Tests passed but target third-party method was NOT covered")
    #         print(f"   Target: {final_state.get('thirdPartyMethod', 'N/A')}")
    #     else:
    #         print("Unknown failure")
    #     print()

    # # Print Maven output if available
    # if last_output:
    #     print("\n=== LAST MAVEN OUTPUT ===")
    #     print(last_output)
    #     print()

    # # Print final Java source
    # print("\n=== FINAL JAVA TEST CODE ===")
    # print(java_source)
    # print()

    # Optional: Save final state to JSON
    if args.output_json:
        output_path = Path(args.output_json)
        output_data = {
            "approved": approved,
            "iteration": iteration,
            "tests_passed": success,
            "target_method_covered": target_covered,
            "coverage_total_lines": coverage_lines,
            "coverage_error": coverage_error,
            "test_file_path": test_rel_path,
            "java_source": java_source,
            "entry_point": final_state.get("entryPoint"),
            "third_party_method": final_state.get("thirdPartyMethod"),
            "trace": final_state.get("trace", []),
        }
        output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
        print(f"Saved final state to: {output_path}")

    # Return appropriate exit code
    return 0 if approved else 1


if __name__ == "__main__":
    raise SystemExit(main())
