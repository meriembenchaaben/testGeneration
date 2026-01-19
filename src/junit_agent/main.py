from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from junit_agent.hf_model import build_local_hf_llm
from junit_agent.deepseek_model import build_deepseek_llm
from junit_agent.graph_app import AppConfig, GenerationInput, build_graph, initial_state


def load_input(json_path: Path) -> GenerationInput:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return GenerationInput(
        entryPoint=data["entryPoint"],
        thirdPartyMethod=data["thirdPartyMethod"],
        directCaller=data.get("directCaller", data["entryPoint"]),
        path=list(data["path"]),
        methodSources=list(data["methodSources"]),
        constructors=list(data.get("constructors", [])),
        setters=list(data.get("setters", [])),
        fieldDeclarations=list(data.get("fieldDeclarations", [])),
        imports=list(data.get("imports", [])),
        testTemplate=data.get("testTemplate", ""),
        conditionCount=data.get("conditionCount", 0),
        callCount=data.get("callCount", 1),
        covered=data.get("covered", False),
        test_package=data.get("testPackage", "generated"),
        test_class_name=data.get("testClassName", "GeneratedReachabilityTest"),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="langgraph-junit-agent")
    p.add_argument("repo_root", type=str, help="Path to the target Maven Java project")
    p.add_argument("input_json", type=str, help="Path to JSON input containing entryPoint/path/methodSources")
    p.add_argument("--mvn", type=str, default="mvn", help="Maven command (default: mvn)")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model id (HuggingFace model or 'deepseek-chat' for DeepSeek)")
    p.add_argument("--api", type=str, choices=["hf", "deepseek"], default="hf", help="API to use: 'hf' for HuggingFace (local) or 'deepseek' for DeepSeek API")
    p.add_argument("--api-key", type=str, help="API key for DeepSeek (or set DEEPSEEK_API_KEY env var)")
    p.add_argument("--iters", type=int, default=5, help="Max generate-run iterations")
    p.add_argument("--temp", type=float, default=0.1, help="Sampling temperature")
    p.add_argument("--max-new", type=int, default=2048, help="Max new tokens")
    p.add_argument("--run-all", action="store_true", help="Run full mvn test suite (default runs only generated test)")
    p.add_argument("--output-json", type=str, help="Optional: Save final state to JSON file")
    p.add_argument("--log-file", type=str, help="Optional: append detailed per-test JSON logs to this file")
    p.add_argument("--all", action="store_true", help="Process all test cases in input file (default: first only)")
    p.add_argument("--resume", type=int, help="Resume from a specific record index (0-based)")
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

def update_json_coverage(json_path: Path, third_party_method: str, direct_caller: str) -> None:
    """
    Update the JSON file to mark all records with matching thirdPartyMethod and directCaller as covered.
    
    Args:
        json_path: Path to the JSON input file
        third_party_method: The third party method that was successfully covered
        direct_caller: The direct caller method
    """
    try:
        print(f"  JSON path: {json_path}")
        print(f"  Looking for thirdPartyMethod: {third_party_method}")
        print(f"  Looking for directCaller: {direct_caller}")
        
        data = json.loads(json_path.read_text(encoding="utf-8"))
        
        updated_count = 0
        
        # Handle both single object and array
        if isinstance(data, dict):
            # Check if it's a single dict with fullMethodsPaths
            if "fullMethodsPaths" in data:
                for path_idx, path_data in enumerate(data["fullMethodsPaths"]):
                    tp_method = path_data.get("thirdPartyMethod")
                    dc = path_data.get("directCaller")
                    if (tp_method == third_party_method and dc == direct_caller):
                        print(f"[DEBUG]   ✓ MATCH FOUND at path {path_idx}")
                        path_data["covered"] = True
                        updated_count += 1
            # Single test case without fullMethodsPaths
            elif data.get("thirdPartyMethod") == third_party_method and data.get("directCaller") == direct_caller:
                data["covered"] = True
                updated_count += 1
        elif isinstance(data, list):
            # Array of test cases
            for idx, tc in enumerate(data):
                # Check nested structure with "fullMethodsPaths"
                if "fullMethodsPaths" in tc:
                    for path_idx, path_data in enumerate(tc["fullMethodsPaths"]):
                        tp_method = path_data.get("thirdPartyMethod")
                        dc = path_data.get("directCaller")
                        if (tp_method == third_party_method and dc == direct_caller):
                            print(f"[DEBUG]   ✓ MATCH FOUND at item {idx}, path {path_idx}")
                            path_data["covered"] = True
                            updated_count += 1
                else:
                    if (tc.get("thirdPartyMethod") == third_party_method and 
                        tc.get("directCaller") == direct_caller):
                        tc["covered"] = True
                        updated_count += 1
        
        if updated_count > 0:
            # Write back to file
            json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"✓ Updated {updated_count} record(s) in JSON file as covered")
            print(f"  Third Party Method: {third_party_method}")
            print(f"  Direct Caller: {direct_caller}")
        else:
            print(f"[DEBUG] No matching records found to update")
    
    except Exception as e:
        print(f"⚠ Warning: Failed to update JSON file: {e}")
        import traceback
        traceback.print_exc()

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
                # Skip if already covered
                if path_data.get("covered", False):
                    continue
                test_template = path_data.get("testTemplate", "")
                pkg, cls = extract_package_and_class(test_template)
                inputs.append(GenerationInput(
                    entryPoint=path_data["entryPoint"],
                    thirdPartyMethod=path_data["thirdPartyMethod"],
                    directCaller=path_data.get("directCaller", path_data["entryPoint"]),
                    path=list(path_data["path"]),
                    methodSources=list(path_data["methodSources"]),
                    constructors=list(path_data.get("constructors", [])),
                    setters=list(path_data.get("setters", [])),
                    fieldDeclarations=list(path_data.get("fieldDeclarations", [])),
                    imports=list(path_data.get("imports", [])),
                    testTemplate=test_template,
                    conditionCount=path_data.get("conditionCount", 0),
                    callCount=path_data.get("callCount", 1),
                    covered=path_data.get("covered", False),
                    test_package=pkg,
                    test_class_name=cls,
                ))
        else:
            # Skip if already covered
            if tc.get("covered", False):
                continue
                
            test_template = tc.get("testTemplate", "")
            pkg, cls = extract_package_and_class(test_template)
            inputs.append(GenerationInput(
                entryPoint=tc["entryPoint"],
                thirdPartyMethod=tc["thirdPartyMethod"],
                directCaller=tc.get("directCaller", tc["entryPoint"]),
                path=list(tc["path"]),
                methodSources=list(tc["methodSources"]),
                constructors=list(tc.get("constructors", [])),
                setters=list(tc.get("setters", [])),
                fieldDeclarations=list(tc.get("fieldDeclarations", [])),
                imports=list(tc.get("imports", [])),
                testTemplate=test_template,
                conditionCount=tc.get("conditionCount", 0),
                callCount=tc.get("callCount", 1),
                covered=tc.get("covered", False),
                test_package=pkg,
                test_class_name=cls,
            ))
    return inputs


def main() -> int:
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    all_test_cases = load_test_cases(Path(args.input_json).resolve())
    
    # Count and report skipped cases
    total_in_file = len(all_test_cases)
    skipped_count = sum(1 for tc in all_test_cases if tc.covered)
    test_cases = [tc for tc in all_test_cases if not tc.covered]
    
    if skipped_count > 0:
        print(f"\n{'='*80}")
        print(f"⊙ Skipped {skipped_count} test case(s) already marked as covered")
        print(f"⊙ Processing {len(test_cases)} uncovered test case(s) from {total_in_file} total")
        print(f"{'='*80}\n")
    
    # Determine which test cases to process
    if args.all:
        # Apply resume offset if specified
        if args.resume is not None:
            if args.resume >= len(test_cases):
                print(f"Error: --resume {args.resume} is beyond the number of available test cases ({len(test_cases)})")
                return 1
            if args.resume < 0:
                print(f"Error: --resume must be a non-negative integer")
                return 1
            print(f"Resuming from record {args.resume} (processing {len(test_cases) - args.resume} test cases)")
            cases_to_process = test_cases[args.resume:]
        else:
            print(f"Processing all {len(test_cases)} test cases from input file")
            cases_to_process = test_cases
    else:
        if args.resume is not None:
            print("Warning: --resume is only applicable with --all flag. Ignoring --resume.")
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

    # Build LLM based on selected API
    if args.api == "deepseek":
        llm = build_deepseek_llm(
            api_key=args.api_key,
            temperature=args.temp,
            max_tokens=args.max_new,
            model=args.model if args.model != "Qwen/Qwen2.5-1.5B-Instruct" else "deepseek-chat",
        )
        print(f"Using DeepSeek API with model: {args.model if args.model != 'Qwen/Qwen2.5-1.5B-Instruct' else 'deepseek-chat'}")
    else:
        llm = build_local_hf_llm(
            model_id=args.model,
            temperature=args.temp,
            max_new_tokens=args.max_new,
        )
        print(f"Using local HuggingFace model: {args.model}")

    app = build_graph(llm=llm, cfg=cfg)
    # Configure logging: console INFO, keep large details for file if requested
    logger = logging.getLogger("junit_agent")
    logger.setLevel(logging.DEBUG)
    # Console handler (INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)

    # Optional text file for detailed iteration logs (prompts, decisions, generated text)
    text_log_fh = None
    text_log_path = Path(args.log_file) if args.log_file else None
    if text_log_path:
        # Open in append mode so multiple runs append
        text_log_fh = open(text_log_path, "a", encoding="utf-8")


        def _tlog(line: str) -> None:
            if text_log_fh is None:
                return
            text_log_fh.write(line + "\n")
            text_log_fh.flush()


        text_log_fh.write("\n" + "=" * 100 + "\n")
        text_log_fh.write(f"LOG SESSION STARTED: {datetime.now().isoformat()}\n")
        text_log_fh.write("=" * 100 + "\n\n")
    
    # Separate file for full prompts (untruncated)
    prompts_log_fh = None
    if text_log_path:
        prompts_log_path = text_log_path.parent / "prompts.log"
        prompts_log_fh = open(prompts_log_path, "a", encoding="utf-8")
        
        def _plog(line: str) -> None:
            if prompts_log_fh is None:
                return
            prompts_log_fh.write(line + "\n")
            prompts_log_fh.flush()
        
        prompts_log_fh.write("\n" + "=" * 100 + "\n")
        prompts_log_fh.write(f"PROMPTS LOG SESSION STARTED: {datetime.now().isoformat()}\n")
        prompts_log_fh.write("=" * 100 + "\n\n")
        

    # Optional JSONL file for detailed per-test logs
    log_fh = None
    json_log_path = None
    if args.log_file:
        # Use .jsonl extension for JSON logs alongside text logs
        base_path = Path(args.log_file)
        json_log_path = base_path.parent / f"{base_path.stem}.jsonl"
        # open append so multiple runs append
        log_fh = open(json_log_path, "a", encoding="utf-8")

    # Store results for all test cases
    all_results = []
    total_approved = 0
    
    # Calculate starting index for display purposes
    start_idx = args.resume if (args.all and args.resume is not None) else 0
    
    # Process each test case
    for idx, inp in enumerate(cases_to_process, start_idx + 1):
        print("\n" + "=" * 80)
        print(f"PROCESSING TEST CASE {idx}/{len(cases_to_process)}")
        print("=" * 80)
        print(f"Entry Point: {inp.entryPoint}")
        print(f"Third Party Method: {inp.thirdPartyMethod}")
        print("=" * 80 + "\n")
        _tlog("=" * 80)
        _tlog(f"PROCESSING TEST CASE {idx}/{len(cases_to_process)}")
        _tlog(f"Entry Point: {inp.entryPoint}")
        _tlog(f"Third Party Method: {inp.thirdPartyMethod}")
        _tlog("=" * 80)

        
        state = initial_state(inp, cfg)

        final_state = None
        logger.info("--- PROCESSING TEST CASE %d/%d ---", idx, len(cases_to_process))
        # Set recursion_limit to allow for max_iterations * nodes_per_iteration
        # Each iteration goes through ~6 nodes (generate, write, validate, run, check_coverage, decide)
        # Add buffer for finalize and ensure 5th iteration completes fully
        recursion_limit = (args.iters * 6) + 5
        for event in app.stream(state, config={"recursion_limit": recursion_limit}):
            for node, st in event.items():
                print(f"[DEBUG-STREAM] Processing node: {node}, has iteration_log: {bool(st.get('iteration_log'))}, approved={st.get('approved', 'NOT_SET')}")
                trace = st.get("trace", [])
                if trace:
                    print(trace[-1])
                    _tlog(trace[-1])  # ✅ write trace to file too

                itlog = st.get("iteration_log", [])
                if not itlog:
                    final_state = st
                    continue

                cur = itlog[-1]
                it = cur.get("iteration", st.get("iteration", "?"))

                if node == "generate":
                    _tlog("-" * 100)
                    _tlog(f"ITERATION {it} — PROMPT")
                    _tlog("-" * 100)
                    prompt_txt = cur.get("prompt", "")
                    _tlog(prompt_txt[:5000])
                    if len(prompt_txt) > 5000:
                        _tlog(f"... (prompt truncated, total {len(prompt_txt)} chars)")
                    
                    # Write full prompt to prompts.log
                    if text_log_path and prompts_log_fh:
                        _plog("=" * 100)
                        _plog(f"TEST CASE {idx}/{len(cases_to_process)} — ITERATION {it}")
                        _plog(f"Entry Point: {inp.entryPoint}")
                        _plog(f"Third Party Method: {inp.thirdPartyMethod}")
                        _plog("=" * 100)
                        _plog(prompt_txt)
                        _plog("")

                    _tlog("-" * 100)
                    _tlog(f"ITERATION {it} — GENERATED JAVA")
                    _tlog("-" * 100)
                    _tlog(cur.get("generated_java", ""))

                elif node == "run":
                    _tlog("-" * 100)
                    _tlog(f"ITERATION {it} — MAVEN RESULT")
                    _tlog("-" * 100)
                    _tlog(f"success={cur.get('maven_success')} exit_code={cur.get('maven_exit_code')}")
                    fb = cur.get("maven_feedback") or ""
                    _tlog(fb[:12000])
                    if len(fb) > 12000:
                        _tlog("... (maven feedback truncated)")

                elif node == "check_coverage":
                    _tlog("-" * 100)
                    _tlog(f"ITERATION {it} — COVERAGE")
                    _tlog("-" * 100)
                    _tlog(json.dumps(cur.get("coverage", {}), indent=2))

                elif node == "decide":
                    _tlog("-" * 100)
                    _tlog(f"ITERATION {it} — DECISION")
                    _tlog("-" * 100)
                    _tlog(f"approved={cur.get('approved')} reason={cur.get('decision_reason')}")

                final_state = st
        
        # Debug: Show which nodes were visited
        print(f"\n[DEBUG] Stream completed. final_state is {'set' if final_state else 'None'}")
        if final_state:
            print(f"[DEBUG] final_state keys: {list(final_state.keys())}")
             

        if not final_state:
            logger.error("No final state produced for test case %d.", idx)
            all_results.append({
                "test_case_index": idx,
                "approved": False,
                "error": "No final state produced"
            })
            continue

        # Extract key results
        approved = bool(final_state.get("approved", False))
        # Debug: print the approved flag
        print(f"[DEBUG] approved (after bool()) = {approved}")
        test_rel_path = final_state.get("test_rel_path", "")
        java_source = final_state.get("java_source", "")
        last_output = final_state.get("last_run_output", "")
        iteration = final_state.get("iteration", 0)
        success = final_state.get("success", False)
        
        # Extract coverage results
        target_covered = final_state.get("target_method_covered", False)
        coverage_lines = final_state.get("coverage_total_lines", 0)
        coverage_error = final_state.get("coverage_error")
        
        # Find the iteration where test became approved (if it did)
        approval_iteration = None
        if approved:
            iteration_log = final_state.get("iteration_log", [])
            for iter_data in iteration_log:
                if iter_data.get("approved"):
                    approval_iteration = iter_data.get("iteration")
                    break

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
            # Print success info with approval iteration
            if approval_iteration is not None:
                print(f"✓ Test approved at iteration {approval_iteration}")
            # Update JSON file to mark this and related records as covered
            try:
                update_json_coverage(
                    Path(args.input_json).resolve(),
                    inp.thirdPartyMethod,
                    inp.directCaller
                )
            except Exception as e:
                print(f"⚠ Warning: Failed to update JSON coverage: {e}")
        
        # Store result
        result_data = {
            "test_case_index": idx,
            "entry_point": inp.entryPoint,
            "third_party_method": inp.thirdPartyMethod,
            "direct_caller": inp.directCaller,
            "call_count": inp.callCount,
            "approved": approved,
            "approval_iteration": approval_iteration,
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

        # Write detailed log for this test case if requested
        if log_fh is not None:
            try:
                detailed = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "test_case_index": idx,
                    "input": {
                        "entryPoint": inp.entryPoint,
                        "thirdPartyMethod": inp.thirdPartyMethod,
                    },
                    "final_state": final_state,
                    "iteration_log": final_state.get("iteration_log", []),
                }
                log_fh.write(json.dumps(detailed, ensure_ascii=False) + "\n")
                log_fh.flush()
            except Exception as e:
                logger.exception("Failed to write to log file: %s", e)

        # Write text log with detailed prompts, decisions, and generated code
        if text_log_fh is not None:
            try:
                text_log_fh.write(f"\n{'='*100}\n")
                text_log_fh.write(f"TEST CASE {idx}: {inp.entryPoint}\n")
                text_log_fh.write(f"{'='*100}\n")
                text_log_fh.write(f"Entry Point: {inp.entryPoint}\n")
                text_log_fh.write(f"Third Party Method: {inp.thirdPartyMethod}\n")
                text_log_fh.write(f"Approved: {approved}\n")
                if approved and approval_iteration is not None:
                    text_log_fh.write(f"Approved at Iteration: {approval_iteration}\n")
                text_log_fh.write(f"Total Iterations: {iteration}\n")
                text_log_fh.write(f"Tests Passed: {success}\n")
                text_log_fh.write(f"Target Method Covered: {target_covered}\n")
                text_log_fh.write(f"\n{'-'*100}\n")
                
                # Write iteration details
                iteration_log = final_state.get("iteration_log", [])
                if iteration_log:
                    text_log_fh.write(f"ITERATION DETAILS ({len(iteration_log)} iterations):\n")
                    text_log_fh.write(f"{'-'*100}\n\n")
                    
                    for iter_data in iteration_log:
                        iter_num = iter_data.get("iteration", "?")
                        text_log_fh.write(f"\n>>> ITERATION {iter_num}\n")
                        text_log_fh.write(f"{'-'*100}\n")
                        
                        # Write prompt
                        prompt = iter_data.get("prompt", "")
                        if prompt:
                            text_log_fh.write(f"\n[PROMPT SENT TO LLM]\n")
                            text_log_fh.write(f"{'-'*50}\n")
                            text_log_fh.write(prompt[:3000])  # Limit to first 3000 chars
                            if len(prompt) > 3000:
                                text_log_fh.write(f"\n... (prompt truncated, total: {len(prompt)} chars)\n")
                            text_log_fh.write(f"\n{'-'*50}\n\n")
                        
                        # Write generated Java code
                        java_code = iter_data.get("generated_java", "")
                        if java_code:
                            text_log_fh.write(f"[GENERATED JAVA CODE]\n")
                            text_log_fh.write(f"{'-'*50}\n")
                            text_log_fh.write(java_code)
                            text_log_fh.write(f"\n{'-'*50}\n\n")
                        
                        # Write Maven feedback
                        maven_feedback = iter_data.get("maven_feedback")
                        if maven_feedback:
                            text_log_fh.write(f"[MAVEN FEEDBACK/DECISION]\n")
                            text_log_fh.write(f"{'-'*50}\n")
                            text_log_fh.write(maven_feedback[:2000])  # Limit to first 2000 chars
                            if len(str(maven_feedback)) > 2000:
                                text_log_fh.write(f"\n... (feedback truncated)\n")
                            text_log_fh.write(f"\n{'-'*50}\n\n")
                        
                        # Write coverage info
                        coverage = iter_data.get("coverage")
                        if coverage:
                            text_log_fh.write(f"[COVERAGE RESULT]\n")
                            text_log_fh.write(f"{'-'*50}\n")
                            text_log_fh.write(f"{json.dumps(coverage, indent=2)}\n")
                            text_log_fh.write(f"{'-'*50}\n\n")
                
                # Final summary for this test case
                text_log_fh.write(f"\n{'='*100}\n")
                text_log_fh.write(f"FINAL RESULT: {'APPROVED' if approved else 'FAILED'}\n")
                if not approved:
                    if not success:
                        text_log_fh.write("Reason: Tests failed to compile or run successfully\n")
                    elif not target_covered:
                        text_log_fh.write(f"Reason: Target method not covered\n")
                        text_log_fh.write(f"Target: {final_state.get('thirdPartyMethod', 'N/A')}\n")
                text_log_fh.write(f"{'='*100}\n\n")
                text_log_fh.flush()
                
            except Exception as e:
                logger.exception("Failed to write text log: %s", e)

    # Print overall summary
    print("\n" + "=" * 80)
    print("=== OVERALL SUMMARY ===")
    print("=" * 80)
    print(f"Total Test Cases: {len(cases_to_process)}")
    print(f"Approved: {total_approved}")
    print(f"Failed: {len(cases_to_process) - total_approved}")
    print(f"Success Rate: {total_approved / len(cases_to_process) * 100:.1f}%")
    print("=" * 80)
    
    # Show approval iterations for successful tests
    if total_approved > 0:
        print("\nApproval Details:")
        for result in all_results:
            if result.get("approved") and result.get("approval_iteration") is not None:
                print(f"  Test {result['test_case_index']}: Approved at iteration {result['approval_iteration']}")
    
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

    # Close file handles
    if log_fh is not None:
        log_fh.close()
    if text_log_fh is not None:
        text_log_fh.close()
    if prompts_log_fh is not None:
        prompts_log_fh.close()

    # Return 0 if all approved, 1 if any failed
    return 0 if total_approved == len(cases_to_process) else 1




if __name__ == "__main__":
    raise SystemExit(main())
