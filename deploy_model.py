import pickle
from flask import Flask, request, jsonify, render_template

# Load the trained model and vectorizer
with open("phishing_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # Render the HTML form

@app.route("/predict", methods=["POST"])
def predict():
    try:
        email_text = request.form["email_text"]  # Get text from the form
        email_vectorized = vectorizer.transform([email_text])  # Convert text to vector
        prediction = model.predict(email_vectorized)[0]  # Make prediction

        result = "Phishing Email 🚨" if prediction == 1 else "Legitimate Email ✅"
        return render_template("index.html", email_text=email_text, result=result)

    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
