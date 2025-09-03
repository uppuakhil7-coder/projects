from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract form data
        form_data = request.form.to_dict()
        print("Received Form Data:", form_data)  # Debugging

        # Convert categorical features
        gender = 1 if form_data.get("gender") == "Male" else 0
        partner = 1 if form_data.get("Partner") == "Yes" else 0
        dependents = 1 if form_data.get("Dependents") == "Yes" else 0
        phone_service = 1 if form_data.get("PhoneService") == "Yes" else 0
        paperless_billing = 1 if form_data.get("PaperlessBilling") == "Yes" else 0

        internet_service = {"DSL": 0, "Fiber optic": 1, "No": 2}.get(form_data.get("InternetService"), 2)
        contract = {"Month-to-month": 0, "One year": 1, "Two year": 2}.get(form_data.get("Contract"), 0)
        payment_method = {"Electronic check": 0, "Mailed check": 1, "Bank transfer": 2, "Credit card": 3}.get(form_data.get("PaymentMethod"), 0)

        numeric_features = [
            float(form_data.get("SeniorCitizen", 0)),
            float(form_data.get("tenure", 0)),
            float(form_data.get("MultipleLines", 0)),
            float(form_data.get("OnlineSecurity", 0)),
            float(form_data.get("OnlineBackup", 0)),
            float(form_data.get("DeviceProtection", 0)),
            float(form_data.get("TechSupport", 0)),
            float(form_data.get("StreamingTV", 0)),
            float(form_data.get("StreamingMovies", 0)),
            float(form_data.get("MonthlyCharges", 0)),
            float(form_data.get("TotalCharges", 0)),
        ]

        processed_features = [
            gender, partner, dependents, phone_service, paperless_billing,
            internet_service, contract, payment_method
        ] + numeric_features

        print("Processed Features:", processed_features)  # Debugging

        input_data = np.array(processed_features).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        result = {"churn": bool(prediction)}

        print("Prediction Result:", result)  # Debugging

        return jsonify(result)

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
