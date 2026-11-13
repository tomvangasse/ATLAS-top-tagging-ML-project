# Top-quark jet tagging on ATLAS Open Data
   (2–3 sentences: the question, and the two models you used)

Does a more accurate top tagger come at the cost of robustness to systematic uncertainties?

I made an XGboost model, and a PyTorch model, and ran them without retraining, on simulated samples. Varying the way the simulated detector works, and how the model predicts events on test data, tests the models' robustness.

## Headline result
   (Small table of R50 and its shift under each variation, plus 3–4 bullets on what they mean, with an accuracy-vs-robustness plot)
| Model | R50 (nominal) | Change under detector variations | Change under shower model |
|---|---|---|---|
| XGBoost | 36.3 | < 0.5% | about +8.7% |
| MLP | 37.8 | < 0.5% | about +5.9% |


## The physics problem
   (What a boosted top jet is, why tagging matters, and what background rejection / R50 means)
   
When colliding particles at near the speed of light, there is a lot of important data to gather from the particle jets that result from the collision. To gain insight from these jets, we need to be able to differentiate between jets that resulted from different particles. This is where jet taggers come in.

I'm testing models on their ability to find top-quark jets, top-quark jet taggers like this are used in searches for new heavy particles that decay to top quarks.

Real world data is possible to use for training tagging models, because the data isn't labelled - we can't tell the model how wrong it is while it's training. This is why physics simulations need to be used to get labelled samples for training. There is a problem with this: **physics simulations can make close predictions of real world events, but won't always be completely accurate to what we measure in real life.**

So, to make sure these models trained on simulated data will work well with real world data, they are tested on more data from simulations that have slight variations, such as:
- Simulation type (Herwig or Pythia 8)
- Detector variations (energy scale, resolution)

This is robustness, if a model performs well tested on varying simulated data, we know it is more likely to perform well with real life data.

### Definitions and explainations

Signal efficiency:
- A top tagger gives a 0-1 range of how much it believes each jet is a top jet
- To use this, you have to set a threshold, above which, you interpret its output as a yes rather than a no
- Choosing the threshold so it keeps 50% of the real top jets, would be 50% signal efficiency

R50 is the background rejection at 50% signal efficiency:
- background rejection = 1/(fraction of background jets accepted)
- An R50 of 36.3 means that 1 in 36.3 background jets pass through at 50% signal efficiency

QCD jets is another name for background jets, it stands for **Quantum Chromodynamics**


## Data
   (The dataset and paper links, which files you used, the 15 features and what they measure, what you excluded and why, training weights)

## Method
### XGBoost baseline (notebooks/02_xgboost.ipynb)
   (max_depth 4, weights, early stopping, SHAP plot)
### Neural network (notebooks/03_mlp.ipynb)
   (Architecture, scaling, loss and optimiser, same split as XGBoost, ROC comparison plot)
### Robustness test (notebooks/04_robustness.ipynb)
   (Table of the variations, why esup/esdown/cer are compared with each other, the noise control from test_nominal_001)

## Results in detail
   (Nominal AUC/R50/R80 table, and the table of R50 changes under each variation)
### What this means
   (Shower modelling dominates, the direction of a shift isn't an improvement, the robustness gap is within the noise)

## Limitations and next steps
   (Subset of the data, noise estimate from one pair of files, only four variations, possible extras)

## Repository layout
   (Folder tree of notebooks/, src/, plots/, results/)

## How to rerun
   (Downloading the data, running the notebooks in order in Colab, dependencies)

## References
   (ATLAS paper, open data record, CERN GitLab example code)