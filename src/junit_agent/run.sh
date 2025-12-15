python -m junit_agent.main /repo /input.json --model Qwen/Qwen2.5-1.5B-Instruct --iters 6 --temp 0.1
python -m junit_agent.main /repo /input.json --run-all   # run full suite instead of only generated test
python -m junit_agent.main /repo /input.json --mvn mvn   # custom mvn binary


# Main Run
python -m junit_agent.main /path/to/maven-repo /path/to/input.json
