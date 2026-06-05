# AI-ENG-201 HW1 Clean Structure

Expected dataset path:

```text
data/titanic3.csv
```

Run experiments from the project root:

```bash
python -m src.experiments
```

Run tests:

```bash
pytest
```

Clean `src/` structure:

```text
src/
  knn.py
  metrics.py
  splits.py
  experiments.py
```

`preprocessing.py` was removed because it is not required by the assignment. The necessary loading, cleaning, imputation, encoding, and EDA helpers are inside `src/experiments.py`.
