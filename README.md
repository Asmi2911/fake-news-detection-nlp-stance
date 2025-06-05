# Fake News Detection via NLP-Based Stance Classification

**A deployable AI system combining semantic NLP, sentiment shifts, and interpretable tree-based models to flag potential misinformation by analyzing headline-article stance alignment.**

## Project Overview

In today's information landscape, distinguishing between legitimate and misleading news is increasingly complex. This project offers a **stance-based fake news detection system** that classifies headline-body pairs as:

- `agree`
- `disagree`
- `discuss`
- `unrelated`

This approach provides **more explainable and granular insight** than simple true/false classification and supports scalable, automated content moderation and policy applications.

## Key AI/NLP Highlights

- **Natural Language Understanding**: Cosine similarity using GloVe and sentence embeddings
- **Polarity & Emotion Detection**: Captures subjectivity shifts across headline and body
- **Linguistic Signals**: TF-IDF overlaps, refuting word count, and token mismatch detection
- **Explainability**: Tree-based models with feature importance make decisions transparent
- **Lightweight AI Pipeline**: Fast and scalable — ideal for real-time deployment

## Dataset

- Used [Fake News Challenge (FNC-1)](https://github.com/FakeNewsChallenge/fnc-1) dataset
- Over 50,000 labeled headline-body pairs
- Balanced across 4 stance classes

## Tools & Technologies

- **Language**: Python
- **Libraries**: `scikit-learn`, `XGBoost`, `LightGBM`, `TextBlob`, `NLTK`, `SentenceTransformers`, `Gensim`
- **NLP Techniques**:
  - TF-IDF vectorization
  - Cosine similarity with GloVe & Sentence Transformers
  - Sentiment analysis using TextBlob
  - Manual feature engineering for refuting indicators

## Methodology

### Feature Engineering
- **Lexical**: TF-IDF, n-gram and token overlap
- **Semantic**: Sentence-transformer embeddings + cosine similarity
- **Emotional**: Polarity and subjectivity shifts
- **Refutation**: Frequency of contradiction keywords like "fake", "fraud", "not"

### Modeling
- Evaluated multiple models:
  - ✅ XGBoost
  - ✅ LightGBM
  - Random Forest
  - Logistic Regression
  - KNN

### Evaluation
- Metrics: Accuracy, F1-score, ROC-AUC
- Validation using stratified splits for generalization
- Confusion matrix and ROC curves plotted for interpretability

## Performance Summary

| Model               | Accuracy | AUC    | Highlights                                   |
|--------------------|----------|--------|----------------------------------------------|
| **XGBoost**         | **89.0%** | **0.996** | Best generalization & feature explainability |
| LightGBM           | 86.0%    | —      | Fast inference for real-time filtering       |
| Random Forest      | 86.3%    | —      | Stable across high variance classes          |
| Logistic Regression| 80.1%    | —      | Useful baseline model                        |
| KNN                | 57.8%    | —      | Underperformed due to sparse feature space   |

> Evaluation visuals (confusion matrix & ROC) included in the notebook.

## Real-World Applications

- **News Platforms**: Filter content for editorial review
- **Social Media**: Flag misleading headlines in early stages
- **Fact-Checking Teams**: Prioritize contradictory content
- **Policy Analysts**: Detect narrative conflicts or agenda shifts
- **LLM Post-Processing**: Guardrails for hallucinated LLM summaries

## Future Scope

- 💡 Integrate **Transformer models (BERT, RoBERTa)** for improved generalization
- 📊 Apply **SMOTE/Focal Loss** to improve performance on minority classes
- 🌐 Build an interactive **Streamlit/Hugging Face Spaces demo**
- 📸 Extend to **multimodal stance detection** (images + text)

> *This project embodies the core of responsible AI — explainability, scalability, and social relevance in tackling misinformation.*
