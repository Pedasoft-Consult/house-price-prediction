from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd


# Load the trained model and scaler
model = joblib.load("models/random_forest_regressor_model.pkl")
scaler = joblib.load("models/scaler.pkl")

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    input_data = pd.DataFrame({
        "area": [float(data["area"])],
        "bedrooms": [int(data["bedrooms"])],
        "bathrooms": [int(data["bathrooms"])],
        "stories": [int(data["stories"])],
        "mainroad": [1 if data["mainroad"] == "yes" else 0],
        "guestroom": [1 if data["guestroom"] == "yes" else 0],
        "basement": [1 if data["basement"] == "yes" else 0],
        "hotwaterheating": [1 if data["hotwaterheating"] == "yes" else 0],
        "airconditioning": [1 if data["airconditioning"] == "yes" else 0],
        "parking": [int(data["parking"])],
        "prefarea": [1 if data["prefarea"] == "yes" else 0],
        "furnishingstatus": [0 if data["furnishingstatus"] == "unfurnished" else (
            1 if data["furnishingstatus"] == "semi-furnished" else 2)]
    })

    # Scale the numerical features
    input_data[["area", "bedrooms", "bathrooms", "stories", "parking"]] = scaler.transform(
        input_data[["area", "bedrooms", "bathrooms", "stories", "parking"]])

    # Predict the house price
    prediction = model.predict(input_data)[0]

    return jsonify({"predicted_price": prediction})


if __name__ == '__main__':
    app.run(debug=True)
