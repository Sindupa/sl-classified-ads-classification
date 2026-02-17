# Sri Lankan Classified Ads Category Classifier

A supervised machine learning project that automatically classifies Sri Lankan online classified advertisements into their correct categories using Natural Language Processing (NLP).

---

## 📌 Project Overview

This project builds a multi-class text classification system capable of predicting advertisement categories from Sinhala, English, or mixed-language ad descriptions.

The system uses TF-IDF vectorization and machine learning algorithms to learn patterns from advertisement data and predict categories.

---

## 🧠 Problem Type

- Supervised Learning  
- Multi-class Classification  
- Natural Language Processing (NLP)

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- LinearSVC (SVM) / Multinomial Naive Bayes
- Flask (Web Application)
- HTML & CSS

---

## 🌐 Web Application

The project includes a Flask-based web application where users can:

- Enter advertisement text
- Get predicted category instantly
- Test Sinhala, English, or mixed input

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
  git clone https://github.com/Sindupa/sl-classified-ads-classification.git
  cd sl-classified-ads-classification
```

### 2. Create Virtual Environment (Recommended)
```bash
  python -m venv venv

  # On Windows
  venv\Scripts\activate
  
  # On macOS/Linux
  source venv/bin/activate
```

### 3. Install Dependencies

```bash
  pip install -r requirements.txt
```

### 4. How to Run the Web Application

```bash
  python app.py

  # Then open your browser and go to:
  http://127.0.0.1:5000
```

---

## 📂 Project Structure

```
sl-classified-ads-classification/
│
├── env/
├── notebooks/
│     └── model.ipynb
│    └── prediction.ipynb (for manual testing)
├── static/
│     └── model/
│       ├── tfidf.pkl
│       ├── model.pkl
│       └── label_encoder.pkl
├── templates/
│   └── index.html
├── app.py
├── README.md
└── requirements.txt
```



