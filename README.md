# Baseline Predictive Pipeline -- ETAI

20231707 - Filipe Caçador

Week 2
LogisticRegression: Train accuracy: 0.679
                    Test accuracy:  0.680
                    F1_score: 0: 0.72
                              1: 0.63

Decision: Train accuracy: 0.679
          Test accuracy:  0.669
                    F1_score: 0: 0.69
                              1: 0.64

Week 3
LogisticRegression: Train accuracy: 0.669
                    Test accuracy:  0.664
                    F1_score: 0: 0.72
                              1: 0.59

Decision Tree: Train accuracy: 0.685
               Test accuracy:  0.683
                    F1_score: 0: 0.72
                              1: 0.63

Week 4
LogisticRegression: Train accuracy: 0.677
                    Test accuracy:  0.677
                    F1_score: 0: 0.71
                              1: 0.63

Decision Tree: Train accuracy: 0.697
               Test accuracy:  0.676
                    F1_score: 0: 0.73
                              1: 0.59

Random Forests: Train accuracy: 0.707
                Test accuracy:  0.700
                    F1_score: 0: 0.73
                              1: 0.66

Week 6
Dummy: Train accuracy: 0.549
       Test accuracy:  0.549
                    F1_score: 0: 0.71
                              1: 0.00

LogisticRegression: Train accuracy: 0.675
                    Test accuracy:  0.674
                    F1_score: 0: 0.74
                              1: 0.57

Decision Tree: Train accuracy: 0.684
               Test accuracy:  0.676
                    F1_score: 0: 0.73
                              1: 0.60

Random Forests: Train accuracy: 0.689
                Test accuracy:  0.677
                    F1_score: 0: 0.72
                              1: 0.61

AdaBoost: Train accuracy: 0.687
          Test accuracy:  0.679
                    F1_score: 0: 0.73
                              1: 0.61

Oblique Random Forests: Train accuracy: 0.689
                        Test accuracy:  0.680
                    F1_score: 0: 0.73
                              1: 0.61

Note on Week 6: "Test accuracy" here is the validation accuracy of nested
The Week  numbers arelower than Week 4's (e.g. Random Forest 0.700 -> 0.677),
but this does not mean the models got worse:
- Week 4 measured "Test accuracy" on a single 80/20 split, on the same test
  rows that were used to compare models week after week. One split of ~1450
  rows varies by about +-0.012 from luck alone, and choosing models by it
  favours whichever happened to score high on those rows.
- Week 6 averages 5 folds and keeps all tuning inside each fold, so the
  estimate is honest (tuning optimism measured at +0.001 to +0.004).
- The data cleaning and features also changed this week (age_cat dropped,
  juv_fel_count rebuilt from juvenile_total, one-hot encoding).
All tuned models score within 0.006 of each other, well inside the fold
standard deviation (~0.013); the corrected paired t-test against Random
Forest gives p > 0.3 for every model, so none is significantly better.

Week 8
Dummy: Train ROC_AUC: 0.500
       Test ROC_AUC:  0.500 +- 0.000
       Test accuracy: 0.549
                    F1_score: 0: 0.71
                              1: 0.00

LogisticRegression: Train ROC_AUC: 0.720
                    Test ROC_AUC:  0.719 +- 0.014
                    Test accuracy: 0.669
                    F1_score: 0: 0.70
                              1: 0.63

Naive Bayes: Train ROC_AUC: 0.706
             Test ROC_AUC:  0.705 +- 0.012
             Test accuracy: 0.621
                    F1_score: 0: 0.73
                              1: 0.37

Decision Tree: Train ROC_AUC: 0.732
               Test ROC_AUC:  0.719 +- 0.013
               Test accuracy: 0.673
                    F1_score: 0: 0.71
                              1: 0.63

Random Forests: Train ROC_AUC: 0.745
                Test ROC_AUC:  0.726 +- 0.015
                Test accuracy: 0.673
                    F1_score: 0: 0.70
                              1: 0.64

AdaBoost: Train ROC_AUC: 0.741
          Test ROC_AUC:  0.728 +- 0.015
          Test accuracy: 0.682
                    F1_score: 0: 0.73
                              1: 0.62

Oblique Random Forests: Train ROC_AUC: 0.746
                        Test ROC_AUC:  0.727 +- 0.015
                        Test accuracy: 0.681
                    F1_score: 0: 0.73
                              1: 0.62

Hist Gradient Boosting: Train ROC_AUC: 0.740
                        Test ROC_AUC:  0.728 +- 0.016
                        Test accuracy: 0.677
                    F1_score: 0: 0.71
                              1: 0.64

Gradient Boosting: Train ROC_AUC: 0.740
                   Test ROC_AUC:  0.729 +- 0.015
                   Test accuracy: 0.682
                    F1_score: 0: 0.73
                              1: 0.62

Note on Week 8: ROC AUC is now the selection metric (what tuning maximises
and runs are ranked by), so "Train" is now train ROC AUC; train accuracy is
no longer measured. As in Week 6, "Test" is the 5-fold validation score of
nested CV (Logistic Regression and Dummy are untuned), on the development set
only; the locked test set is still unused. Accuracy and F1 use a 0.5 threshold.
- ROC AUC measures how well the model ranks people by risk, at every
  threshold at once, so it does not depend on the 0.5 cut-off. Accuracy
  hides this: Naive Bayes has 0.62 accuracy but only 0.24 recall for
  reoffenders at 0.5 (F1 of 0.37), even though its ranking is close to the others.
- Every run also reports balanced accuracy, precision, recall, F1, F2 and
  average precision (see `results/run_*.txt`).
- New models: Naive Bayes, Gradient Boosting, Hist Gradient Boosting.
- The five ensembles score within 0.003 ROC AUC of each other. The corrected
  paired t-test against Gradient Boosting (best, 0.729) gives p > 0.15 for
  all of them, so none is significantly better. Logistic Regression
  (-0.010, p = 0.015) and Naive Bayes (-0.023, p = 0.003) are significantly
  lower; Decision Tree is borderline (-0.010, p = 0.096).


This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
