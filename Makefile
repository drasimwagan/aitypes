.PHONY: all figures paper check clean

all: figures paper

figures:
	cd code && python3 figures.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

check:
	cd code && python3 sweep_check.py

clean:
	cd paper && latexmk -c
