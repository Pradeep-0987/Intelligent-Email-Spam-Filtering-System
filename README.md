# 🛡️ Intelligent Email Spam Filtering System

A machine learning-powered web application that detects spam emails and classifies messages in real-time, featuring an interactive dashboard and model analytics.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📸 Features

- 🔍 **Real-time classification** — paste any email or message and get instant spam/ham verdict
- 📊 **Model analytics dashboard** — accuracy, precision, recall, F1 score, confusion matrix
- 🤖 **3-model comparison** — Naive Bayes vs Logistic Regression vs Linear SVM
- 📋 **Session history** — track all messages analyzed in the current session
- ⚡ **Quick test examples** — built-in spam and ham examples for instant demo
- 🎨 **Modern dark UI** — clean, professional, recruiter-ready interface

---

## 🧠 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **Linear SVM** ✅ | **99.1%** | **97.9%** | **95.3%** | **96.6%** |
| Naive Bayes | 99.0% | 99.3% | 93.3% | 96.2% |
| Logistic Regression | 98.9% | 98.6% | 93.3% | 95.9% |

> **Linear SVM** selected as the production model based on best F1 score.

---

## 🗂️ Project Structure
Intelligent-Email-Spam-Filtering-System/
│
├── app.py                  # Streamlit web application
├── train.py                # Model training script
├── requirements.txt        # Python dependencies
│
├── data/
│   └── spam.tsv            # UCI SMS Spam Collection (5,572 messages)
│
├── models/
│   ├── spam_model.pkl      # Trained model pipeline
│   └── metrics.json        # Evaluation metrics for all models
│
└── README.md

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Pradeep-0987/Intelligent-Email-Spam-Filtering-System.git
cd Intelligent-Email-Spam-Filtering-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (optional — pre-trained model included)

```bash
python train.py
```

### 4. Launch the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🔧 How It Works
Raw Email
│
▼
Text Preprocessing
• Lowercase
• Replace numbers → NUM
• Replace URLs → URL
• Remove punctuation
│
▼
TF-IDF Vectorization
• Unigrams + Bigrams
• 10,000 features
• Sublinear TF scaling
│
▼
Linear SVM Classifier
│
▼
SPAM / HAM + Confidence Score

---

## 📁 Dataset

**UCI SMS Spam Collection**  
5,572 real SMS messages collected for spam research.

| Class | Count | Percentage |
|-------|-------|------------|
| Ham (Legitimate) | 4,825 | 86.6% |
| Spam | 747 | 13.4% |

---

## 🧪 Example Predictions

| Message | Prediction | Confidence |
|---------|-----------|------------|
| "WINNER!! You've been selected for a cash prize of £1000..." | 🚨 SPAM | 99.2% |
| "FREE entry to win FA Cup Final tickets. Text FA to 87121" | 🚨 SPAM | 98.7% |
| "Hey, are we still on for lunch tomorrow?" | ✅ HAM | 97.4% |
| "Please review the attached report before the meeting." | ✅ HAM | 96.8% |

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Scikit-learn** — ML pipeline, TF-IDF, SVM, Naive Bayes, Logistic Regression
- **Streamlit** — interactive web interface
- **Pandas & NumPy** — data processing
- **Joblib** — model serialization

---

## 👤 Author

**R Pradeep Kumar **  
AIML Undergrad · Sri Krishna Institution of Technology, Bengaluru  

[![GitHub](https://img.shields.io/badge/GitHub-Pradeep--0987-181717?style=flat&logo=github)](https://github.com/Pradeep-0987)
[![Email](https://img.shields.io/badge/Email-kumarrpradeep970%40gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:kumarrpradeep970@gmail.com)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
