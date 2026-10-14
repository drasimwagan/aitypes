.PHONY: all figures paper check clean

# Override on systems where the interpreter is called `python`: make PYTHON=python figures
PYTHON ?= python3

all: figures paper

figures:
	cd code && $(PYTHON) figures.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

check:
	cd code && $(PYTHON) sweep_check.py

clean:
	cd paper && latexmk -c
