# DSAI4205-Sentiment-Analysis

## Project Overview
[cite_start]This repository contains my contributions to a group project for the Big Data Analytics course (DSAI4205) at The Hong Kong Polytechnic University[cite: 3, 9, 10]. [cite_start]The project's objective was to perform a comprehensive sentiment analysis on a dataset of 3,125 Reddit comments discussing the Israeli-Palestinian conflict, classifying them as neutral, positive, or negative[cite: 23]. 

[cite_start]Because this is a highly polarized and sensitive topic, the linguistic expression often includes sarcasm, indirect criticism, historical analogies, and coded language, posing significant challenges for standard Natural Language Processing (NLP)[cite: 55, 56]. 

[cite_start]Our team evaluated a baseline Logistic Regression model using TF-IDF against five advanced hybrid models[cite: 25, 26]. [cite_start]**The Convolutional Neural Network (CNN) paired with GloVe word embeddings—which I developed—achieved the highest accuracy (70.5%) among all models tested**.

## My Contributions
This repository specifically highlights my individual technical contributions to the broader group project:

### 1. Data Cleaning Pipeline (`Data_cleaning_DSAI.ipynb`)
*Co-authored with one group member.*
* [cite_start]**Noise Reduction:** Developed a custom cleaning pipeline to unescape HTML, remove URLs, and strip markdown artifacts[cite: 172].
* [cite_start]**Standardization:** Converted text to lowercase, removed non-alphanumeric characters, and expanded standard contractions (e.g., "won't" to "will not") to reduce vocabulary noise[cite: 174, 176].
* [cite_start]**Tokenization & Lemmatization:** Applied NLTK's WordNet lemmatizer and filtered out ultra-short tokens[cite: 177].
* [cite_start]**Outlier Handling:** Removed spam, single-word emojis, and extreme outliers based on word-count distribution boundaries[cite: 170, 171]. 
* [cite_start]**Result:** Reduced the dataset to a high-quality corpus of 3,032 standardized comments for downstream modeling[cite: 24].

### 2. Advanced Modeling: GloVe + CNN (`DSAI4205_Project_CNN_GloVe.ipynb`)
*Sole Author.*
* [cite_start]**Architecture:** Built a Convolutional Neural Network (CNN) combined with pre-trained GloVe 300-dimensional word embeddings (`glove-wiki-gigaword-300`)[cite: 268, 271]. 
* [cite_start]**Feature Extraction:** Utilized a Conv1D layer with 128 filters to scan comments and detect 5-gram sentiment patterns, followed by GlobalMaxPooling1D to preserve the strongest signals[cite: 276, 277].
* [cite_start]**Domain Adaptation:** Set the embedding layer as trainable, allowing the model to fine-tune the general-purpose GloVe vectors to the specific vocabulary and context of the Reddit discussions[cite: 273, 296].
* [cite_start]**Performance:** This model successfully captured local patterns and n-gram features (e.g., "ceasefire now") that traditional bag-of-words models missed[cite: 269]. [cite_start]It outperformed the TF-IDF baseline and all other advanced models (including BERT+XGBoost and Word2Vec+BiLSTM)[cite: 27, 28].

## Model Performance Comparison
[cite_start]Below is a summary of how my GloVe + CNN model stacked up against the project's baseline and other advanced models[cite: 458].

| Model Architecture | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **GloVe + CNN (My Model)** | **0.705** | **0.722** | **0.718** | **0.720** |
| Word2Vec + BiLSTM | 0.6517 | 0.6461 | 0.6305 | 0.6353 |
| Baseline (TF-IDF + LR) | 0.6343 | 0.6549 | 0.6343 | 0.6220 |
| BERT + XGBoost | 0.575 | 0.57 | 0.57 | 0.56 |
| FastText + Random Forest | 0.5237 | 0.4856 | 0.5237 | 0.4520 |
| GloVe + Complement Naive Bayes | 0.4729 | 0.5133 | 0.4729 | 0.3559 |

## Repository Structure
* `Data_cleaning_DSAI.ipynb`: Jupyter notebook containing the full exploratory data analysis (EDA) and text preprocessing pipeline.
* `DSAI4205_Project_CNN_GloVe.ipynb`: Jupyter notebook containing the tokenization, embedding matrix construction, CNN architecture, and model evaluation.
* `Final Report.pdf`: The comprehensive group report detailing the methodology, baseline establishment, and comparative analysis of all models.

## How to Run
1. Clone this repository.
2. Ensure you have the required libraries installed (`pandas`, `numpy`, `nltk`, `seaborn`, `matplotlib`, `tensorflow`, `gensim`, `scikit-learn`).
3. Run the `Data_cleaning_DSAI.ipynb` notebook first to generate the cleaned CSV file.
4. Run the `DSAI4205_Project_CNN_GloVe.ipynb` notebook, ensuring the file path to the cleaned CSV is correctly mapped in the data-loading cell. (Note: The GloVe 300d embeddings require a ~376MB download upon first execution).
