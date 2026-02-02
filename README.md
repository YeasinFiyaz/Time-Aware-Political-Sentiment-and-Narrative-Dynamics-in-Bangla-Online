## Time-Aware Political Sentiment and Narrative Dynamics in Bangla Online Media

# About the Project

This repository contains a complete machine learning pipeline for analyzing political sentiment and narrative dynamics over time using Bangla (Bengali) online textual data.
The study focuses on understanding how public political attitudes evolve across different time periods, combining supervised sentiment classification, topic modeling, and temporal trend analysis.

# Research Motivation

Political narratives and public sentiment are not static; they evolve in response to changing socio-political contexts.
This project aims to:

Quantify sentiment shifts (positive, neutral, negative) over time

Identify dominant political discussion topics

Analyze temporal dynamics of sentiment and narratives

Provide empirical evidence using Bangla language data, which remains under-explored in computational social science

Bangladesh is treated as a case study, but the framework is language- and region-agnostic.

# Methodology Overview

The pipeline follows a modular and interpretable machine learning approach:

1️⃣ Data Preparation

Cleaning and normalization of Bangla text

Label harmonization across datasets

Removal of noise and empty records

2️⃣ Sentiment Classification (Supervised)

Models implemented and compared:

- Logistic Regression (TF-IDF)
- Linear Support Vector Machine
- Stochastic Gradient Descent (Logistic loss)
- Multinomial Naïve Bayes

Evaluation metrics:

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC curves (one-vs-rest)

3️⃣ Topic Modeling (Unsupervised)

- Latent Dirichlet Allocation (LDA)
- Topic-wise word distributions
- Topic vs. sentiment interaction analysis

4️⃣ Temporal Analysis

- Yearly sentiment share trends
- Monthly net sentiment index
- Time-aware visualization of political discourse evolution

📊 Outputs

The pipeline automatically generates:

📈 Learning curves & validation curves

📊 Model comparison plots

🔁 Confusion matrices & ROC curves

🗂 Topic distribution tables

⏱ Temporal sentiment trend plots

All figures used in the paper are reproducible from code.

# Project Structure
.
├── src/                # All scripts (data prep, training, analysis)
├── figures/            # Final plots used in paper/thesis
├── outputs/            # Generated CSVs and intermediate results
├── data/               # Raw datasets (not publicly tracked if restricted)
├── requirements.txt    # Python dependencies
└── README.md

# How to Run
python src/00_check_setup.py
python src/01_prepare_datasets.py
python src/02_train_baseline.py
python src/03_apply_to_banglamedia.py
python src/04_topic_modeling_lda.py
python src/06_time_trend_sentiment.py

# Reproducibility
- Fixed random seeds where applicable
- Standardized preprocessing pipeline
- Clear separation of training, evaluation, and analysis
- Compatible with Python 3.13+
This repository is suitable for:

- MSc / BSc thesis projects
- Computational social science research
- NLP research in low-resource languages
- Political communication analysis
- Time-aware sentiment studies


## Author
# Yeasin Fiyaz
Dept of CSE, Brac University

Research interests:
Machine Learning · NLP · Medical Image Processing · Data-Driven Policy Analysis

# Built using:

- Python
- scikit-learn
- pandas
- matplotlib

Department of Computer Science & Engineering
Research interests:
Machine Learning · Bioinformatics · Social Computi · Data-Driven Policy Analysis
