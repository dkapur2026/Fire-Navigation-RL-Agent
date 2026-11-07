.PHONY: install test test-fast demo results audit clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest -q

test-fast:
	python -m pytest -q -m "not slow"

demo:  ## baseline vs oracle GIFs for the README
	emberpath run --profile single --novelty door_jam --agent baseline --seed 1 --gif docs/media/door_jam_baseline.gif --quiet
	emberpath run --profile single --novelty door_jam --agent oracle   --seed 1 --gif docs/media/door_jam_oracle.gif --quiet

results:  ## regenerate docs/results.md (about 4 minutes)
	python scripts/make_results.py --seeds 60

audit:  ## try to hack the reward with degenerate policies (about 20 seconds)
	python scripts/audit_reward.py --seeds 20

clean:
	rm -rf .pytest_cache build *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
