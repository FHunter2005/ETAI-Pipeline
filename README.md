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

Week 9
Dummy: Train ROC_AUC: 0.500
       Test ROC_AUC:  0.500 +- 0.000
       Test accuracy: 0.549
                    F1_score: 0: 0.71
                              1: 0.00

LogisticRegression: Train ROC_AUC: 0.717
                    Test ROC_AUC:  0.716 +- 0.011
                    Test accuracy: 0.667
                    F1_score: 0: 0.69
                              1: 0.64

Naive Bayes: Train ROC_AUC: 0.709
             Test ROC_AUC:  0.708 +- 0.013
             Test accuracy: 0.631
                    F1_score: 0: 0.73
                              1: 0.41

Decision Tree: Train ROC_AUC: 0.733
               Test ROC_AUC:  0.718 +- 0.014
               Test accuracy: 0.670
                    F1_score: 0: 0.71
                              1: 0.62

Random Forests: Train ROC_AUC: 0.737
                Test ROC_AUC:  0.724 +- 0.013
                Test accuracy: 0.669
                    F1_score: 0: 0.70
                              1: 0.63

AdaBoost: Train ROC_AUC: 0.735
          Test ROC_AUC:  0.723 +- 0.013
          Test accuracy: 0.679
                    F1_score: 0: 0.73
                              1: 0.61

Oblique Random Forests: Train ROC_AUC: 0.738
                        Test ROC_AUC:  0.724 +- 0.012
                        Test accuracy: 0.678
                    F1_score: 0: 0.73
                              1: 0.61

Hist Gradient Boosting: Train ROC_AUC: 0.733
                        Test ROC_AUC:  0.725 +- 0.013
                        Test accuracy: 0.673
                    F1_score: 0: 0.70
                              1: 0.65

Gradient Boosting: Train ROC_AUC: 0.731
                   Test ROC_AUC:  0.726 +- 0.013
                   Test accuracy: 0.681
                    F1_score: 0: 0.73
                              1: 0.61

XGBoost: Train ROC_AUC: 0.731
         Test ROC_AUC:  0.725 +- 0.012
         Test accuracy: 0.680
                    F1_score: 0: 0.73
                              1: 0.62

Note on Week 9 (same setup as Week 8; locked test set still unused):
- New: XGBoost and a feature-selection step (a feature is kept if enough of
  a random forest, gradient boosting and an L1 logistic regression vote for
  it). It kept only `age` and `priors_count` in every model, and scores
  barely moved (about -0.003 vs Week 8), so the other features add little.
- Best model: Gradient Boosting (0.726). It has the highest ROC AUC and
  accuracy (0.681) of all models, and the smallest train-test gap (+0.005)
  and lowest FPR for African-American defendants (0.29) of the ensembles.
  The other ensembles are tied with it (p > 0.2); Decision Tree, Logistic
  Regression and Naive Bayes are significantly worse (p < 0.01).
- Fairness: every model has a higher FPR for African-American than for
  Caucasian defendants (Gradient Boosting 0.29 vs 0.16), but a smaller gap
  than COMPAS's own score (0.45 vs 0.23).


Week 10 (class notebook 09, "Beyond COMPAS", option A: Adult income)
New dataset, same code: `python main.py config_adult.yaml`. COMPAS (`python main.py`)
gives exactly the same numbers as in Week 9. Details in the "Beyond COMPAS" section below.

XGBoost (default hyperparameters): Train ROC_AUC: 0.940
                                   Test ROC_AUC:  0.913 +- 0.016
                                   Test accuracy: 0.864
                    F1_score: 0: 0.91
                              1: 0.69


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
├── main.py                # entry point: run the whole pipeline (python main.py [config file])
├── compare_runs.py         # corrected paired t-test between the runs of a scoreboard
├── config.yaml             # all tunable settings live here (COMPAS)
├── config_adult.yaml       # the same pipeline on the Adult income data
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   ├── tuning.py           # Optuna tuning + nested cross-validation
│   └── results.py          # saves each run's report, scoreboard and predictions to disk
├── results/                # created automatically -- one file per run (not tracked in git)
├── results_adult/          # same, for config_adult.yaml (not tracked in git)
├── predictions/
│   └── adult_new_predictions.csv   # id, probability, prediction for data/adult_new.csv
└── data/
    ├── compas_two_year_recidivism.csv
    ├── README.md            # problem description + full data dictionary
    ├── adult_train.csv, adult_new.csv
    └── ADULT_README.md      # Adult column descriptions
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 10 (notebook 09) | Beyond COMPAS, option A: new data | The pipeline runs on a second dataset (Adult income) with its own config: `main.py` takes the config file as an argument, placeholder cleaning covers every text column, the fairness report no longer needs COMPAS's score, and the final model predicts a label-free file (`predictions/adult_new_predictions.csv`). See "Beyond COMPAS" below |

## Beyond COMPAS (option A: Adult income)

Same pipeline, new data: does this person earn more than 50,000 USD a year?
(`data/adult_train.csv`, 6,000 people, 24% earn more; see `data/ADULT_README.md`.)

### How to run

```bash
python main.py                      # COMPAS, as before (config.yaml)
python main.py config_adult.yaml    # Adult income
python compare_runs.py config_adult.yaml   # corrected paired t-test between the runs in the scoreboard
```

`config_adult.yaml` writes its reports and scoreboard to `results_adult/`, and the predictions
for the 2,000 new people (`data/adult_new.csv`) to `predictions/adult_new_predictions.csv`
(2,000 rows in, 2,000 rows out: `id`, `probability`, `prediction`).

### Choices in the config

- **Sensitive attribute:** `sex` is the one audited (never a feature). `race` is also left out of
  the features: it is the other sensitive attribute and should not drive the prediction.
  `relationship` (Husband / Wife) still carries part of the information in `sex`.
- **Not features:** `id`, `fnlwgt` (a sampling weight, not a characteristic of the person) and
  `education` (same information as `education_num`, which keeps the order).
- **Missing values:** `?` becomes a real missing value. `occupation` gets a `_was_missing` flag,
  because the people with unknown workclass / occupation earn > 50K half as often.
  `native_country` and `v01`-`v06` are only imputed.
- **Validity rules:** Adult's own (`age` 17-90, `hours_per_week` 1-99, capital columns >= 0).
  `capital_gain` = 99999 is the census top code, so it stays.
- **Encoder:** target encoding, so `native_country` (41 values, many rare) stays one column.
- **`v01`-`v25`:** all kept as candidate features; feature selection decides.

### What broke, why, and how it was fixed

| What broke | Why | Fix (works for both configs) |
|---|---|---|
| `python main.py config_adult.yaml` still ran COMPAS | `main()` always loaded `config.yaml` | `main.py` reads the config path from the command line (default `config.yaml`) |
| `KeyError: 'score_text'` in the fairness report | `fairness_report()` and `split_features_target()` named COMPAS's score column | The reference score is now optional and set in the config (`data.reference_score`: column, positive values, label). COMPAS sets it, Adult doesn't |
| `?` stayed as a category in `workclass`, `occupation` and `native_country` (no error) | `_canonicalize_categories()` only touched the columns listed in `canonical_categories`, and Adult lists none | Placeholder tokens and extra spaces are cleaned in every text column; `canonical_categories` only adds the spelling maps. Check: 379 / 381 / 86 missing after cleaning |
| 74 valid people would lose their age | COMPAS's rule `age: {min: 18}` copied to Adult | Adult has its own `validity_rules` (config only) |
| Reports talked about reoffending and `config.yaml` | Text written for COMPAS | General wording ("true negatives predicted positive"), and the config file name comes from the argument |
| No way to predict people without a target | The pipeline stopped at the refit | `predict_new_data()` in `main.py`: same cleaning and feature split, no rows dropped, saved by `id` (runs when the config has `data.new_path`) |
| Runs of both problems in one scoreboard | One `output` folder | Each config has its own `output` paths |

After these changes, `python main.py` on COMPAS gives exactly the same scores, confusion matrix,
fairness table and features kept as before (checked with tuning off and with tuning on).

### Results

5-fold cross-validation on the development set (4,800 rows; the locked test set is unused),
feature selection with `min_votes: 2`. Tuned runs are nested CV with 30 trials.
p-value: corrected paired t-test against the best run, on the same folds.

| Model | Tuned | ROC AUC | Accuracy | F1 (class 1) | p vs best |
|---|---|---|---|---|---|
| **Target** | | **>= 0.90** | **>= 0.855** | **>= 0.66** | |
| Dummy (no model) | no | 0.500 | 0.76 | 0.00 | |
| **XGBoost** (final) | no | **0.913 +- 0.016** | **0.864** | **0.686** | best |
| Hist Gradient Boosting | yes | 0.912 +- 0.016 | 0.865 | 0.684 | 0.27 |
| Gradient Boosting | no | 0.912 +- 0.016 | 0.862 | 0.676 | 0.36 |
| Gradient Boosting | yes | 0.909 +- 0.013 | 0.859 | 0.665 | 0.19 |
| XGBoost | yes | 0.908 +- 0.015 | 0.857 | 0.669 | 0.10 |
| XGBoost, no feature selection | no | 0.905 +- 0.018 | 0.858 | 0.663 | 0.12 |
| XGBoost, `scale_pos_weight: 3.2` | no | 0.911 +- 0.016 | 0.815 | 0.683 | 0.006 |
| Hist Gradient Boosting | no | 0.903 +- 0.016 | 0.856 | 0.667 | 0.010 |
| AdaBoost | no | 0.903 +- 0.016 | 0.850 | 0.653 | 0.001 |
| Logistic Regression | no | 0.896 +- 0.015 | 0.847 | 0.640 | 0.004 |
| Oblique Random Forest | no | 0.896 +- 0.016 | 0.854 | 0.659 | < 0.001 |
| Random Forest | no | 0.891 +- 0.015 | 0.841 | 0.633 | < 0.001 |
| Naive Bayes | no | 0.876 +- 0.013 | 0.801 | 0.433 | < 0.001 |
| Decision Tree | no | 0.725 +- 0.024 | 0.803 | 0.583 | < 0.001 |

- **Final model:** XGBoost with the hyperparameters in `config_adult.yaml` (tuning off). It
  reaches the three targets in the same run. The tuned boosting models reach them too, but
  tuning did not improve on the defaults (differences inside the fold noise, p >= 0.10), so the
  simpler run is kept.
- **Features kept (8 of 36):** `age`, `education_num`, `capital_gain`, `capital_loss`,
  `hours_per_week`, `marital_status`, `occupation`, `relationship`. **No `v` column is kept**,
  and neither are `workclass`, `native_country` or the `occupation_was_missing` flag.
- **Fairness (sex):** false positive rate 0.08 for men and 0.03 for women.
- **Predictions:** 2,000 rows, 19.4% predicted to earn > 50K.

The hardest assumption to find was the `?` placeholder, because nothing crashed: the cleaning
ran, the model trained, and `?` was simply treated as one more category. Only the
`value_counts(dropna=False)` check showed it. The fairness report was the opposite: it crashed
at once, so it was easy to find. What got the pipeline to the targets was the model family
(only boosted trees pass 0.90 ROC AUC) together with feature selection, which removes the 25
undocumented `v` columns (0.905 -> 0.913 for XGBoost). Weighting the classes shows why the
three targets are asked together: it raises recall from 0.62 to 0.83 but drops accuracy to
0.815, below the target.

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
