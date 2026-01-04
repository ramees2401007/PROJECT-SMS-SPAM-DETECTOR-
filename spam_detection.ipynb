# ========================================================
# PROJECT: Email Spam Detection using Supervised Learning
# STUDENT: Muhamed Ramees
# ROLL NO: AA.SC.P2MCA2401007
# ========================================================

# --- 1. IMPORT LIBRARIES ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 2. LOAD DATASET ---
# Using the SMS Spam Collection dataset
# Ensure 'spam.csv' is in the same directory
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: 'spam.csv' not found. Please upload the dataset.")

# Data Cleaning: Keep only necessary columns and rename them
df = df[['v1', 'v2']]
df.columns = ['Label', 'Message']

# Convert Labels to Numbers (Spam=1, Ham=0)
df['Label_Num'] = df['Label'].map({'ham': 0, 'spam': 1})

print("Dataset Preview:")
print(df.head())
print(f"\nTotal Messages: {len(df)}")

# --- 3. DATA PREPROCESSING (NLP) ---
# Your report promised Tokenization and Stop-word removal. 
# TfidfVectorizer performs these steps automatically.

# Split Data: 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(
    df['Message'], df['Label_Num'], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

print(f"\nVocabulary size: {len(tfidf.get_feature_names_out())}")

# --- 4. MODEL IMPLEMENTATION ---

# Model A: Naive Bayes (Promised in Proposal)
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# Model B: Logistic Regression (Promised in Proposal)
lr_model = LogisticRegression()
lr_model.fit(X_train_vec, y_train)

# --- 5. EVALUATION & METRICS ---

def evaluate_model(model, name):
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"\n--- {name} Results ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, preds))
    
    # Plot Confusion Matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Run Evaluation
evaluate_model(nb_model, "Naive Bayes")
evaluate_model(lr_model, "Logistic Regression")

# --- 6. LIVE PREDICTION TEST ---
print("\n--- SYSTEM TEST ---")
sample_spam = ["URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt WORD to 81010"]
sample_ham = ["Hey, are we still going for lunch today?"]

# Function to predict
def predict_message(message):
    vec = tfidf.transform(message)
    pred = nb_model.predict(vec)
    return "SPAM" if pred[0] == 1 else "NOT SPAM"

print(f"Message: {sample_spam[0]}\nPredicted: {predict_message(sample_spam)}")
print(f"Message: {sample_ham[0]}\nPredicted: {predict_message(sample_ham)}")
