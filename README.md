# Replication Package: Mutant Realism in Deep Learning

This repository is the **replication package** for the paper:

> *An Empirical Study of the Realism of Mutants in Deep Learning*  
> Zaheed Ahmed, Philip Makedonski, Jens Grabowski

The repository should be read **in conjunction with the paper**.

---

## Contents of the Replication Package

```
scripts/
  execution_matrix.py
  killing_probability.py
  quantify_realism.py
  model_validator.py
  visualization.ipynb

results/
  execution_matrix/
  killing_probability/
  coupling_strength/
  detectability_overlap/
```

- `scripts/` contains the analysis code used in the study.
- `results/` contains the **final CSV result files** used to generate all figures reported in the paper.

Intermediate artifacts (trained models, execution matrices per run, and killing probabilities) are not included.

---

## Reproducibility Scope

Training the deep learning models used in this study is computationally expensive and may take weeks.  
For this reason, this replication package **does not support rerunning model training**.

Instead, it provides:
- the complete analysis pipeline, and
- the final CSV artifacts required to regenerate all reported figures.

All boxplots and bar charts in the paper can be regenerated from the CSV files in `results/` using the visualization script.

---

## Regenerating Figures

To regenerate the figures reported in the paper:

1. Open and run:
   ```
   scripts/visualization.ipynb
   ```
2. The script reads the CSV files from `results/` and produces the corresponding figures.

---

## Datasets and Trained Models

The curated datasets, instrumented files, and the trained models used in the study are **not included** in this repository due to size considerations.

They will be made available in a **separate repository once the paper is accepted**, ensuring a stable and final version of the experimental artifacts.

The current repository already provides all scripts and final result artifacts required to inspect and reproduce the reported analyses without retraining.

---
