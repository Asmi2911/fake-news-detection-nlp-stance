# Fake News Detection via NLP & Stance Classification

In today's digital age, the rapid dissemination of information has made distinguishing between genuine news and misinformation increasingly challenging. This project presents an AI-driven approach to detect potential fake news by analyzing the **stance** between a news headline and its corresponding article body—categorizing their relationship as *agree*, *disagree*, *discuss*, or *unrelated*.

## Project Overview

- **Objective**: Develop a stance detection model to identify inconsistencies between news headlines and article bodies, aiding in the detection of misleading or false information.
- **Dataset**: Utilized the [Fake News Challenge (FNC-1)](https://github.com/FakeNewsChallenge/fnc-1) dataset, comprising over 50,000 headline-body pairs labeled across four stance categories.
- **Approach**: Employed a combination of natural language processing techniques and machine learning models to analyze semantic relationships and classify stances effectively.

## Methodology

### Feature Engineering

- **Lexical Features**: Implemented TF-IDF vectors, n-gram overlaps, and word overlap measures to capture surface-level textual similarities.
- **Semantic Features**: Leveraged GloVe embeddings and sentence transformers to compute cosine similarity, capturing deeper semantic relationships.
- **Sentiment Analysis**: Analyzed polarity and subjectivity differences between headlines and article bodies to detect emotional and subjective discrepancies.
- **Refuting Indicators**: Identified presence of refuting words (e.g., "hoax", "fake", "fraud") that often signal disagreement or misinformation.

### Modeling Techniques

- **Algorithms Used**:
  - XGBoost
  - LightGBM
  - Random Forest
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
- **Evaluation Metrics**: Assessed models using accuracy, precision, recall, F1-score, and ROC-AUC to ensure robust performance across all stance categories.

## Results

- **Best Model**: XGBoost achieved an accuracy of approximately 89% and an ROC-AUC of 0.996 on the FNC-1 test set.
- **Performance Highlights**:
  - High precision in distinguishing "unrelated" stances.
  - Effective detection of subtle disagreements and discussions, which are often challenging to classify.
  - Demonstrated robustness across all four stance categories, indicating a well-generalized model.

## Real-World Applications

- **Media Monitoring**: Assists news organizations in flagging potentially misleading articles for editorial review.
- **Social Media Platforms**: Enhances content moderation by identifying posts that contradict their headlines, a common trait in clickbait or fake news.
- **Fact-Checking Organizations**: Streamlines the process of verifying news by highlighting articles with conflicting headline-body stances.
- **Policy Makers**: Provides insights into the spread of misinformation, aiding in the development of countermeasures.

## Contributions

This project was developed encompassing the following responsibilities:

- **Data Preprocessing**: Cleaned and prepared the FNC-1 dataset for analysis.
- **Feature Engineering**: Designed and implemented both lexical and semantic features to capture nuanced textual relationships.
- **Model Development**: Trained and fine-tuned multiple machine learning models, selecting the most effective based on performance metrics.
- **Evaluation & Analysis**: Conducted thorough evaluations, including confusion matrices and ROC curves, to assess model efficacy.
- **Documentation**: Compiled comprehensive documentation to facilitate understanding and potential future enhancements.

## 🚀 Future Enhancements

- **Addressing Class Imbalance**: Implement techniques like SMOTE or focal loss to improve model performance on minority classes.
- **Incorporating Transformer Models**: Explore advanced models like BERT or RoBERTa for potentially improved semantic understanding.
- **Real-Time Deployment**: Develop a user-friendly interface using frameworks like Streamlit for real-time stance detection.
- **Multimodal Analysis**: Extend the model to analyze multimedia content, such as images or videos, alongside text for a more holistic fake news detection system.

This project reflects a broader interest in building practical, explainable AI systems that address societal challenges like misinformation — with applications in journalism, policy, and digital safety.

