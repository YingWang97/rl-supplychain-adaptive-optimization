.PHONY: lint test train eval

lint:
	flake8

test:
	pytest -q

train:
	python scripts/train.py +experiment=toy_dqn --algo dqn --episodes 5

eval:
	python scripts/eval.py --run_dir runs/toy_dqn
