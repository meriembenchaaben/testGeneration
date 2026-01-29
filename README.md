# LangGraph JUnit Agent

LangGraph-powered loop that generates a minimal JUnit 5 *reachability* test for a Maven Java project, runs it, and iterates until the test compiles/runs and the requested third‑party method is reached (based on JaCoCo coverage).

## How it works
Given a JSON description of a call chain:

1) **Load input**: reads `entryPoint`, `thirdPartyMethod`, the expected `path`, and supporting `methodSources` (+ optional constructors/setters/fields/imports/template).
2) **Generate**: prompts an LLM to output a single Java file (JUnit 5) whose only goal is to execute the call chain and invoke the target third‑party method (no assertions).
3) **Write**: saves the generated test under `src/test/java/...` in the target Maven project.
4) **Run**: executes Maven tests (either only the generated test class, or the full suite).
5) **Check coverage**: parses JaCoCo output to determine whether the target method was executed.
6) **Loop**: if the build fails or coverage is missing, Maven feedback + coverage info are fed back to the model and steps 2–5 repeat until success or `--iters` is reached.

## Install
```bash
pip install -e .
```

## Run

### 1) Create & activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Run the agent (local Hugging Face model)
```bash
python -m junit_agent.main \
  /path/to/maven-project \
  input.json \
  --log-file agent.log
```

### 3) Run the agent (DeepSeek API)
```bash
python -m junit_agent.main /path/to/maven-project input.json \
  --api deepseek \
  --api-key "your-deepseek-api-key-here" \
  --model deepseek-chat \
  --log-file agent.log
```

You can also set `DEEPSEEK_API_KEY` instead of passing `--api-key`.

## CLI options

All backends share the same CLI; `--api` selects the model source.

For the authoritative list, run:
```bash
python -m junit_agent.main --help
```

Key options:

```bash
--api {hf,deepseek}   Model backend (default: hf)
--model MODEL_ID      HF model id (default: Qwen/Qwen2.5-1.5B-Instruct) or DeepSeek model name
--api-key KEY         DeepSeek API key (or set DEEPSEEK_API_KEY)
--temp FLOAT          Sampling temperature (default: 0.1)
--max-new N           Max new tokens (default: 2048)
--iters N             Max generate→run iterations (default: 5)
--run-all             Run the full Maven test suite (default: only the generated test)
--mvn CMD             Maven command (default: mvn)
--all                 Process all test cases from the input file (default: first only)
--resume IDX          Resume from record index (0-based). Only applies with --all
--log-file PATH       Write a readable log plus a sidecar .jsonl file (and prompts.log)
--output-json PATH    Save the final state to JSON
```

Notes:
- If input records have `covered: true`, they are skipped.
- When multiple records would produce the same test class name, the runner automatically deduplicates by appending a number.

## Input format (JSON)

The agent accepts:

- a single object, e.g. `{ "entryPoint": ..., "thirdPartyMethod": ..., ... }`, or
- a list of objects, or
- an object/list containing `fullMethodsPaths`, where each element is treated as a separate test case.

Common fields per test case:

- `entryPoint` (string): fully qualified method where execution must begin
- `thirdPartyMethod` (string): fully qualified third‑party method that should be invoked
- `directCaller` (string, optional): defaults to `entryPoint`
- `path` (array): ordered list of methods that must be traversed
- `methodSources` (array): source code strings for methods in the chain
- Optional context: `constructors`, `setters`, `fieldDeclarations`, `imports`, `testTemplate`
- Optional output control: `testPackage`, `testClassName`
- `covered` (bool, optional): if true, the agent will skip that record

## LangGraph execution flow

    ┌───────────┐
    │  Generate │  ← LLM creates / fixes JUnit test
    └─────┬─────┘
      │
    ┌─────▼─────┐
    │   Write   │  ← Write test file into Maven repo
    └─────┬─────┘
      │
    ┌─────▼─────┐
    │    Run    │  ← Run Maven tests
    └─────┬─────┘
      │
    ┌─────▼─────────┐
    │ Check Coverage│  ← Verify target method is reached
    └─────┬─────────┘
      │
    ┌─────▼─────┐
    │  Decide   │  ← Approve or retry
    └─────┬─────┘
      │
    ┌─────▼─────┐
    │ Finalize  │  ← Success or failure
    └───────────┘
