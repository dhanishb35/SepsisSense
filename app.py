import numpy as np
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        data = [
            float(request.form["hr"]),
            float(request.form["o2sat"]),
            float(request.form["temp"]),
            float(request.form["sbp"]),
            float(request.form["map"]),
            float(request.form["dbp"]),
            float(request.form["resp"]),
            float(request.form["baseexcess"]),
            float(request.form["fio2"]),
            float(request.form["ph"]),
            float(request.form["paco2"]),
            float(request.form["glucose"]),
            float(request.form["potassium"]),
            float(request.form["hct"]),
            float(request.form["age"]),
            float(request.form["gender"]),
            float(request.form["unit1"]),
            float(request.form["unit2"]),
            float(request.form["hospadmtime"]),
            float(request.form["iculos"])
        ]

        data = np.array(data).reshape(1, -1)
        data = scaler.transform(data)

        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1] * 100

        if prediction == 1:
            risk = "🔴 High Risk"
            prediction_text = "Sepsis Detected"
            recommendation = "Immediate clinical evaluation is recommended."
        else:
            risk = "🟢 Low Risk"
            prediction_text = "No Sepsis Detected"
            recommendation = "Continue routine monitoring."

        return render_template(
            "result.html",
            risk=risk,
            prediction=prediction_text,
            probability=round(probability, 2),
            recommendation=recommendation
        )

    return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)