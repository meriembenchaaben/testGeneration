from __future__ import annotations

import argparse
import json
import re
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
        constructors=list(data.get("constructors", [])),
        setters=list(data.get("setters", [])),
        getters=list(data.get("getters", [])),
        imports=list(data.get("imports", [])),
        testTemplate=data.get("testTemplate", ""),
        conditionCount=data.get("conditionCount", 0),
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
    p.add_argument("--all", action="store_true", help="Process all test cases in input file (default: first only)")
    return p.parse_args()

def extract_package_and_class(test_template: str) -> tuple[str, str]:
    """
    Extract package name and class name from test template.
    Returns:
        tuple: (package_name, class_name)
    """
    package_name = "generated"
    class_name = "GeneratedReachabilityTest"
    
    lines = test_template.split('\n')
    for line in lines:
        line = line.strip()
        
        if line.startswith('package '):
            package_name = line.replace('package ', '').replace(';', '').strip()
        
        if 'class ' in line and not line.startswith('//'):
            match = re.search(r'\bclass\s+(\w+)', line)
            if match:
                class_name = match.group(1)
    
    return package_name, class_name

def load_test_cases(json_path: Path) -> list[GenerationInput]:
    """Load test cases from JSON file. Supports both single object and array of objects."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    
    # Check if it's a single test case or array
    if isinstance(data, dict):
        # Single test case - wrap in array
        test_cases = [data]
    elif isinstance(data, list):
        # Array of test cases
        test_cases = data
    else:
        raise ValueError("JSON must be either a single object or an array of objects")
    
    # Convert to GenerationInput objects
    inputs = []
    for idx, tc in enumerate(test_cases):
        # Handle nested structure with "fullMethodsPaths"
        if "fullMethodsPaths" in tc:
            for path_data in tc["fullMethodsPaths"]:
                test_template = path_data.get("testTemplate", "")
                pkg, cls = extract_package_and_class(test_template)
                inputs.append(GenerationInput(
                    entryPoint=path_data["entryPoint"],
                    thirdPartyMethod=path_data["thirdPartyMethod"],
                    path=list(path_data["path"]),
                    methodSources=list(path_data["methodSources"]),
                    constructors=list(path_data.get("constructors", [])),
                    setters=list(path_data.get("setters", [])),
                    getters=list(path_data.get("getters", [])),
                    imports=list(path_data.get("imports", [])),
                    testTemplate=test_template,
                    conditionCount=path_data.get("conditionCount", 0),
                    test_package=pkg,
                    test_class_name=cls,
                ))
        else:
            test_template = tc.get("testTemplate", "")
            pkg, cls = extract_package_and_class(test_template)
            inputs.append(GenerationInput(
                entryPoint=tc["entryPoint"],
                thirdPartyMethod=tc["thirdPartyMethod"],
                path=list(tc["path"]),
                methodSources=list(tc["methodSources"]),
                constructors=list(tc.get("constructors", [])),
                setters=list(tc.get("setters", [])),
                getters=list(tc.get("getters", [])),
                imports=list(tc.get("imports", [])),
                testTemplate=test_template,
                conditionCount=tc.get("conditionCount", 0),
                test_package=pkg,
                test_class_name=cls,
            ))
    return inputs


def main() -> int:
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    test_cases = load_test_cases(Path(args.input_json).resolve())
    
    # Determine which test cases to process
    if args.all:
        print(f"Processing all {len(test_cases)} test cases from input file")
        cases_to_process = test_cases
    else:
        if len(test_cases) > 1:
            print(f"Found {len(test_cases)} test cases in input file")
            print("Processing first test case only. Use --all to process all cases.")
        cases_to_process = [test_cases[0]]
    
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
    
    # Store results for all test cases
    all_results = []
    total_approved = 0
    
    # Process each test case
    for idx, inp in enumerate(cases_to_process, 1):
        print("\n" + "=" * 80)
        print(f"PROCESSING TEST CASE {idx}/{len(cases_to_process)}")
        print("=" * 80)
        print(f"Entry Point: {inp.entryPoint}")
        print(f"Third Party Method: {inp.thirdPartyMethod}")
        print("=" * 80 + "\n")
        
        state = initial_state(inp, cfg)

        final_state = None
        for event in app.stream(state):
            for node, st in event.items():
                trace = st.get("trace", [])
                if trace:
                    print(trace[-1])
                final_state = st

        if not final_state:
            print(f"No final state produced for test case {idx}.")
            all_results.append({
                "test_case_index": idx,
                "approved": False,
                "error": "No final state produced"
            })
            continue

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

        # Print failure reason if not approved
        if not approved:
            print("\n=== FAILURE REASON ===")
            if not success:
                print("Tests failed to compile or run successfully")
            elif not target_covered:
                print("Tests passed but target third-party method was NOT covered")
                print(f"   Target: {final_state.get('thirdPartyMethod', 'N/A')}")
            else:
                print("Unknown failure")
            print()
        else:
            total_approved += 1
        
        # Store result
        result_data = {
            "test_case_index": idx,
            "entry_point": inp.entryPoint,
            "third_party_method": inp.thirdPartyMethod,
            "approved": approved,
            "iteration": iteration,
            "tests_passed": success,
            "target_method_covered": target_covered,
            "coverage_total_lines": coverage_lines,
            "coverage_error": coverage_error,
            "test_file_path": test_rel_path,
            "java_source": java_source,
            "trace": final_state.get("trace", []),
        }
        all_results.append(result_data)

    # Print overall summary
    print("\n" + "=" * 80)
    print("=== OVERALL SUMMARY ===")
    print("=" * 80)
    print(f"Total Test Cases: {len(cases_to_process)}")
    print(f"Approved: {total_approved}")
    print(f"Failed: {len(cases_to_process) - total_approved}")
    print(f"Success Rate: {total_approved / len(cases_to_process) * 100:.1f}%")
    print("=" * 80 + "\n")

    # Optional: Save all results to JSON
    if args.output_json:
        output_path = Path(args.output_json)
        output_data = {
            "total_cases": len(cases_to_process),
            "approved": total_approved,
            "failed": len(cases_to_process) - total_approved,
            "success_rate": total_approved / len(cases_to_process),
            "results": all_results
        }
        output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
        print(f"✓ Saved all results to: {output_path}")

    # Return 0 if all approved, 1 if any failed
    return 0 if total_approved == len(cases_to_process) else 1

if __name__ == "__main__":
    raise SystemExit(main())
