# README.md

## What this does
A LangGraph-powered loop that:
1) reads JSON input describing an entry point, call path, and a target third-party method
2) uses a local open-source Hugging Face model to generate a minimal JUnit 5 test (single Java file)
3) writes the test into a Maven repo
4) runs Maven (optionally only the generated test)
5) feeds Maven output back to the model and repeats until success or max iterations
6) streams every step to the console so you see what the agent is doing.

## Install
```bash
pip install -e .
```
## Run


### 1. Create & activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```
### Run the agent
```bash
python -m junit_agent.main \
  /path/to/maven-project \
  input.json \
  --log-file agent.log
```

### Common options

```bash
--model MODEL_ID        Hugging Face model (default: Qwen/Qwen2.5-1.5B-Instruct)
--iters N               Max generate→run iterations (default: 5)
--run-all               Run full Maven test suite (default: generated test only)
--log-file PATH         Enable detailed human-readable + JSONL logs
```

### LangGraph Execution Flow

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
