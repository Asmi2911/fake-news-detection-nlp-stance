
# Fake News Detection Using NLP & Stance Classification

In a digital world increasingly shaped by headlines and viral content, misinformation has emerged as a serious threat to public trust and democratic discourse. This project presents a scalable AI solution that leverages natural language processing to identify inconsistencies between a headline and its corresponding article — a reliable proxy for detecting fake news.

Rather than relying on binary classification, this model applies **stance detection** to classify each headline-body pair as one of four nuanced relationships: **agree**, **disagree**, **discuss**, or **unrelated**. This allows for deeper linguistic analysis and better interpretability, particularly in cases where content is misleading rather than entirely false.

## Why Stance Detection?

Traditional fake news classifiers often fall short when articles bend the truth subtly. By focusing on **semantic alignment** between headlines and article bodies, this project captures the emotional, tonal, and factual mismatches that are common in misleading content. For instance, an article body subtly refuting a sensational headline can be flagged based on a **"disagree"** stance.

## Techniques and Tools

- **Dataset**: [FNC-1: Fake News Challenge](https://github.com/FakeNewsChallenge/fnc-1)  
- **Feature Engineering**:
  - GloVe and Sentence Transformer embeddings
  - Cosine similarity, refuting word count
  - Polarity and subjectivity differences
  - TF-IDF and n-gram overlaps
- **Models**:
  - XGBoost (top performer)
  - LightGBM
  - Random Forest
  - Logistic Regression (baseline)
- **Validation**: Performance validated on official FNC-1 test set

## 🎯 Results

- **Accuracy**: ~89% with XGBoost
- **AUC (XGBoost)**: 0.996
- **Robust across all stance classes**, with particularly strong results on "unrelated" and "discuss" categories
- **High generalizability**: External test set confirmed consistency

## 📊 Business Relevance

This system can support:
- **News platforms** in pre-screening articles for editorial review
- **Social media companies** in prioritizing moderation queues
- **Fact-checking organizations** by clustering articles with high stance inconsistency
- **Policy watchdogs** monitoring shifts in public discourse and media framing

By automating early detection of suspicious content, this approach helps reduce manual triage effort, enhances platform credibility, and strengthens public resilience to misinformation.

## 👩‍💻 Role

This project is curated and restructured for:
- Designing and implementing the NLP pipeline
- Engineering semantic and lexical features
- Training and tuning multiple classification models
- Interpreting results through explainable AI techniques
- Analyzing real-world business applications and deployment feasibility

This project is an ongoing effort to explore practical, explainable, and impactful solutions to the misinformation problem — with the goal of building a more trustworthy information ecosystem.

