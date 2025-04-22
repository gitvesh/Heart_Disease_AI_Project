import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model_path = "model/heart_disease_model.pkl"

# ✅ Load the model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Train and save the model first.")

with open(model_path, "rb") as model_file:
    model = joblib.load(model_file)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Extract features from form
            features = [
                float(request.form["age"]),
                float(request.form["sex"]),
                float(request.form["cp"]),
                float(request.form["trestbps"]),
                float(request.form["chol"]),
                float(request.form["fbs"]),
                float(request.form["restecg"]),
                float(request.form["thalach"]),
                float(request.form["exang"]),
                float(request.form["oldpeak"]),
                float(request.form["slope"]),
                float(request.form["ca"]),
                float(request.form["thal"]),
            ]

            features = np.array(features).reshape(1, -1)
            prediction = model.predict(features)[0]

            # Redirect to result page
            return render_template("result.html", prediction=prediction, form_data=request.form)

        except Exception as e:
            return render_template("result.html", prediction=f"Error: {str(e)}", form_data={})

    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
