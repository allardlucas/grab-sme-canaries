PYTHON ?= python3

.PHONY: eval test

eval:
	$(PYTHON) eval/run_eval.py

test:
	$(PYTHON) -m unittest discover -s tests -v
