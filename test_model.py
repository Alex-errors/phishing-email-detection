import pickle
import numpy as np

# Load the trained model and vectorizer with correct filenames
with open("phishing_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Test with a sample email
sample_email = ["Dear user, your account has been compromised. Click the link to reset your password."]
X_sample = vectorizer.transform(sample_email)

# Predict
prediction = model.predict(X_sample)

# Output the result
print("Phishing Email" if prediction[0] == 1 else "Legitimate Email")
