# 📧 Spam Message Classifier using Multinomial Naive Bayes

This project focuses on building an accurate and efficient spam filtering system using the **Multinomial Naive Bayes classifier**, achieving a baseline accuracy of **96%**, and improving to **98%** with hyperparameter tuning using **GridSearchCV**.

---

## 🚀 Project Highlights

- ✅ **Accuracy (Base Model)**: 96%
- 🔧 **Accuracy (After GridSearchCV)**: 98%
- 🔤 **Text Preprocessing**:
  - Lowercasing
  - Stopword Removal
  - Punctuation Removal
  - Stemming
  - Lemmatization
- ✨ **Feature Extraction**: TF-IDF Vectorization
- 🧪 **Train-Test Split**: 80/20
- ☁️ **Visualizations**: Word Clouds for Spam and Ham

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- NLTK / spaCy
- Pandas / NumPy
- Matplotlib / WordCloud

---

## 🧹 Preprocessing Pipeline

Before training the model, raw SMS texts are cleaned and normalized using the following preprocessing techniques:

### 🔽 Lowercasing
All text is converted to lowercase to maintain uniformity and reduce redundancy in token representation (e.g., "Spam" and "spam" are treated the same).

### ❌ Stopword Removal
Common English words like "the", "is", "and", etc., are removed as they provide little value in determining spam vs. ham.

### 🔡 Punctuation Removal
All punctuation marks are stripped from the text to reduce noise and focus only on the actual words.

### 🌱 Stemming
Words are reduced to their base or root form using a rule-based approach (e.g., "running", "runner" → "run"). This helps in dimensionality reduction.

### 🧠 Lemmatization
Unlike stemming, lemmatization uses linguistic knowledge to bring words to their meaningful base form (e.g., "better" → "good"). This ensures more accurate normalization.

### 🧮 TF-IDF Vectorization
Each message is transformed into a numeric vector based on term frequency and inverse document frequency. This highlights important words and downplays common ones.
