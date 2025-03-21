import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
import string
import re

# Download stopwords for text processing
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv("phishing_email.csv")

# Check for missing values
print("Checking for missing values...")
print(df.isnull().sum())

# Remove missing values (if any)
df = df.dropna()

# Define a function to clean email text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove extra spaces
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])  # Remove stopwords
    return text

# Apply text cleaning function
print(df.head())  # Print the first few rows
print(df.columns)  # Print the column names
df['text_combined'] = df['text_combined'].apply(clean_text)

# Encode labels (Convert 'phishing' → 1 and 'not_phishing' → 0)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split data into training and testing sets (80% train, 20% test)
print(df.columns)  # This will print all column names
print(df.head())   # This will show the first few rows
X_train, X_test, y_train, y_test = train_test_split(df['text_combined'], df['label'], test_size=0.2, random_state=42)

# Convert text to numerical format using TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save the processed data and vectorizer for later use
import pickle
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("processed_data.pkl", "wb") as f:
    pickle.dump((X_train_tfidf, X_test_tfidf, y_train, y_test), f)

print("Data preprocessing completed successfully!")
df.to_csv('preprocessed_email.csv', index=False)
