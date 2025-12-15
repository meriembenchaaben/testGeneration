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
