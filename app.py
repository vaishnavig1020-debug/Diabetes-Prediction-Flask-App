from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("diabetes_predictor.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))  # If using scaler

@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/symptoms")
def symptoms():
    return render_template("symptoms.html")

@app.route("/tips")
def tips():
    return render_template("tips.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Read form input
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform([data])  # Scale input
    output = model.predict(final_input)[0]

    if output == 1:
        result = "⚠ You are likely to have diabetes."
    else:
        result = "✅ You are unlikely to have diabetes."

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)

