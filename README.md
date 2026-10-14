# Types of Intelligence

**Adaptation Spectra, Consolidation, and the Partial Order of Minds**

Asim Imdad Wagan · aiwagan@gmail.com · working draft v0.2 (October 2026)

Paper: [`paper/main.pdf`](paper/main.pdf)

## Summary

Debates about whether AI is "smarter" than humans assume intelligence is a single number. This paper argues it is not. Any adaptive system — a bacterium, an ant colony, a human, a language model — is described by a **profile**: what it can learn at each timescale, how much of what it learns quickly survives into slow memory (*consolidation*), how many bodies share one memory (*instance topology*), how far its goals reach, and how strongly it acts on its world.

On this view, humans and today's deployed language models differ less in whether they learn in real time (both do, within an episode) than in **where learning is consolidated**: in the individual, or in the model lineage, in batches, with a lag.

In a minimal, exactly solvable model of a drifting world we prove that:

- a **consolidating individual** (human-like) and an **amnesic but far more experienced lineage** (LLM-like) are *incomparable*: each wins in worlds that drift at different rates;
- the crossover drift rate falls with the **square of the lineage's staleness** and **saturates in the amount of pooled experience** — more data cannot substitute for consolidation;
- a **hive learner** that consolidates continuously across many instances dominates both.

Intelligence is therefore a partial order, and any scalar measure implicitly chooses which timescales matter.

## Repository layout

```
paper/            LaTeX source (main.tex, references.bib), compiled main.pdf
paper/figures/    Figures 1-3 (PDF) and numerical_check.json
code/model.py     closed-form results (Props. 1-2, Eq. 7) and the Monte Carlo simulator
code/figures.py   regenerates Figures 1-3 and the analytic-vs-Monte-Carlo table
code/sweep_check.py  randomised check of Theorem 1, Proposition 3 and the Q* approximation
code/check_long.py   long Monte Carlo runs for the lineage agents
```

## Reproducing

Requires Python 3 with the packages in `requirements.txt`, and a TeX distribution with `latexmk` (pdflatex + BibTeX).

```bash
pip install -r requirements.txt
make figures   # ~90 s: regenerates paper/figures/*.pdf and numerical_check.json
make paper     # builds paper/main.pdf
make check     # randomised verification of the theorems (400 parameter settings)
```

Random seeds: 7 (Figure 2 regret curves), 11 (analytic vs. Monte Carlo table), 21–23 (long runs), 0 (randomised check).

## Citing

See [`CITATION.cff`](CITATION.cff).

## Use of AI

This work was developed in an extended dialogue between the author and Claude (Anthropic), which contributed to the literature search, formalisation, proofs, simulation code, figures and drafting. The author reviewed all content and takes full responsibility for it.
