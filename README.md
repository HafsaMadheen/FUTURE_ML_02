# AI Support Ticket Classification System

## Future Interns – Machine Learning Task 2

An AI-powered NLP application that automatically classifies customer support tickets into categories and predicts ticket priority levels using Machine Learning.

---

## Features

### NLP Pipeline

- Text Cleaning
- Lowercase Conversion
- Remove Numbers
- Remove Punctuation
- Stopword Removal
- Lemmatization
- TF-IDF Vectorization

### Machine Learning Models

- Logistic Regression
- Multinomial Naive Bayes
- Random Forest

### Predictions

- Ticket Category Classification
- Ticket Priority Prediction
- Confidence Score

### Dashboard Features

- Dark AI Theme
- Glassmorphism UI
- Responsive Design
- Graph Analytics
- Real-time Predictions

---

## Dataset

Customer Support Ticket Dataset:

https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset

---

## Project Structure

```text
SUPPORT_TICKET_CLASSIFIER/
│
├── app.py
├── train_model.py
├── generate_graphs.py
├── requirements.txt
├── README.md
│
├── models/
├── static/
├── templates/
└── dataset/
```

## Installation

### Clone Repository

```bash
git clone <repository_url>
cd SUPPORT_TICKET_CLASSIFIER
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train Models

```bash

```python train_model.py

---

## Generate Graphs

```bash
python generate_graphs.py
```

---

## Run Flask App

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## Generated Visualizations

- Category Distribution
- Priority Distribution
- Word Cloud
- Category Confusion Matrix
- Priority Confusion Matrix
- Model Comparison
- Feature Importance

---

## Future Improvements

- Deep Learning Models
- BERT Classification
- Live Database Integration
- Ticket Assignment Automation
- Email Alert System
- Cloud Deployment

---

## Author

Future Interns Task 2 Submission

AI Support Ticket Classification System

Machine Learning + NLP + Flask Dashboard