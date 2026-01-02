# Using DeepSeek API

## Installation

First, install the updated dependencies:

```bash
pip install -e .
```

Or install just the new dependency:

```bash
pip install langchain-openai
```

## Usage

### Option 1: Using Environment Variable

Set your DeepSeek API key as an environment variable:

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key-here"
```

Then run the command with `--api deepseek`:

```bash
python -m junit_agent.main /u/gamageyo/fika/experiments/pdfbox/fontbox third_party_apis_full_methods.json \
  --api deepseek \
  --model deepseek-chat \
  --log-file agent.log
```

### Option 2: Using Command-Line Argument

Pass the API key directly via `--api-key`:

```bash
python -m junit_agent.main /u/gamageyo/fika/experiments/pdfbox/fontbox third_party_apis_full_methods.json \
  --api deepseek \
  --api-key "your-deepseek-api-key-here" \
  --model deepseek-chat \
  --log-file agent.log
```

## Available Models

For DeepSeek, you can use:
- `deepseek-chat` (default) - DeepSeek V3

## Command-Line Options

- `--api`: Choose API backend (`hf` for HuggingFace local, `deepseek` for DeepSeek API)
- `--api-key`: DeepSeek API key (alternative to DEEPSEEK_API_KEY env var)
- `--model`: Model name (e.g., `deepseek-chat`)
- `--temp`: Temperature (default: 0.1)
- `--max-new`: Max tokens (default: 2048)
- `--iters`: Max iterations (default: 5)
- `--log-file`: Log file path
- `--all`: Process all test cases in input file

## Example: Full Command

```bash
export DEEPSEEK_API_KEY="sk-your-key-here"

python -m junit_agent.main \
  /u/gamageyo/fika/experiments/pdfbox/fontbox \
  third_party_apis_full_methods.json \
  --api deepseek \
  --model deepseek-chat \
  --log-file agent.log \
  --temp 0.1 \
  --max-new 2048 \
  --iters 5
```

## Backward Compatibility

The original HuggingFace local model usage still works (this is the default):

```bash
CUDA_VISIBLE_DEVICES=2,3 python -m junit_agent.main \
  /u/gamageyo/fika/experiments/pdfbox/fontbox \
  third_party_apis_full_methods.json \
  --log-file agent.log \
  --model meta-llama/Llama-3.3-70B-Instruct
```

Or explicitly specify `--api hf`:

```bash
python -m junit_agent.main /path/to/repo input.json --api hf --model meta-llama/Llama-3.3-70B-Instruct
```
