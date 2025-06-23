# Fake-News Stance Detection 
### Classical ML pipeline • Intel®-Optimised

[![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)](https://python.org)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Intel](https://img.shields.io/badge/Optimised%20for-Intel®%20CPUs-lightgrey?logo=intel)](https://www.intel.com/content/www/us/en/developer/topic-technology/ai.html)

> **Task**&nbsp; &nbsp; Given a headline + article body, classify the stance as  
> `agree • disagree • discuss • unrelated`.


## Dataset

| File | Description |
|------|-------------|
| `train_bodies.csv`  | News body texts (training) |
| `train_stances.csv` | Ground-truth stance labels |
| `competition_test_bodies.csv` / `competition_test_stances.csv` | Held-out test split |
| `scorer.py` | Official FNC-1 scoring script (patched → UTF-8) |


## Environment (Conda)

```bash
# Create Intel-optimised env
conda create -n intel-fake-news python=3.9 -y
conda activate intel-fake-news

# Core libs + Intel® Extension for Scikit-learn
conda install -c conda-forge \
      scikit-learn-intelex numpy pandas matplotlib seaborn \
      lightgbm xgboost nltk wordcloud jupyterlab -y
The environment ships Intel® oneAPI-powered NumPy / SciPy and the Intel® Extension for Scikit-learn.

## Results

| Model         |  Accuracy | Macro F1 | Train Time | Intel Gain\*          |
| ------------- | --------: | -------: | ---------: | --------------------- |
| **XGBoost**   | **0.926** | **0.68** | **5.68 s** | OpenMP + MKL          |
| LightGBM      |     0.916 |     0.64 |     3.94 s | Threading             |
| Random Forest |     0.913 |     0.64 |    21.26 s | Patched (`sklearnex`) |
| K-NN          |     0.732 |     0.42 |     0.20 s | Vectorised distance   |
| Logistic Reg. |     0.746 |     0.26 |     0.50 s | DAAL4py accelerated   |
* Speed-up measured against the same code without patch_sklearn() on an Intel® Core™ i7-13700H CPU.

## Pipeline Overview
Pre-processing – tokenisation, stop-word removal, stemming/lemmatisation

Feature Engineering – TF-IDF & CountVectorizer text vectors

Model zoo – LR • RF • KNN • XGBoost • LightGBM (all Intel-ready)

Evaluation – scorer.py (FNC-1) + classification_report

Visuals – Seaborn class distributions, word clouds


## Real-World Applications

- **News Platforms**: Filter content for editorial review
- **Social Media**: Flag misleading headlines in early stages
- **Fact-Checking Teams**: Prioritize contradictory content
- **Policy Analysts**: Detect narrative conflicts or agenda shifts
- **LLM Post-Processing**: Guardrails for hallucinated LLM summaries

## Future Scope

| Idea                                                               | Benefit                            |
| ------------------------------------------------------------------ | ---------------------------------- |
| Replace TF-IDF + classical ML with DistilBERT via Optimum-Intel | Modern embeddings + oneAPI kernels |
| Apply Intel® Neural Compressor for post-training quantisation      | Smaller/faster models for edge     |
| Deploy a Flask API on Intel® DevCloud                              | Real-time stance inference demo    |

> *This project embodies the core of responsible AI — explainability, scalability, and social relevance in tackling misinformation.*
