# House Price Prediction Web App

## Overview
This project is a **machine learning-powered web application** for predicting house prices based on various features such as area, bedrooms, bathrooms, and more. It uses **Flask** for the backend and **HTML/CSS/JavaScript** for the frontend.

## Features
- **Machine Learning Models:** Trained multiple regression models, including Linear Regression, Random Forest, Gradient Boosting, and SVR.
- **Web-Based User Interface:** Users can input house details through an intuitive web form.
- **Live Predictions:** The app predicts house prices in real-time using a trained model.
- **Visualization & Logs:** Includes feature importance plots and logs for performance tracking.

## Technologies Used
- **Python, Flask** (Backend API)
- **Scikit-Learn, Pandas, NumPy** (Machine Learning)
- **HTML, CSS, JavaScript** (Frontend)
- **Matplotlib, Seaborn** (Visualization)
- **Joblib** (Model Storage)

## Project Structure
```
├── models/                  # Trained machine learning models & scaler
├── visualization/           # Plots and charts
├── predictions/             # Sample predictions
├── logs/                    # Model performance logs
├── templates/               # HTML templates for Flask
│   ├── index.html           # Main user interface
├── app.py                   # Flask application
├── housing.csv              # Dataset
├── README.md                # Project documentation
```

## Installation
### Prerequisites
Make sure you have **Python 3.7+** installed.

### Steps to Run the Application
1. **Clone the repository:**
   ```sh
   git clone https://github.com/Pedasoft-Consult/house-price-prediction.git
   cd house-price-prediction
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Flask application:**
   ```sh
   python app.py
   ```

5. **Access the web app**
   Open a browser and go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Usage
1. **Fill in the house details** in the form.
2. **Click the "Predict Price" button.**
3. The **predicted house price** will be displayed.

## Model Training & Evaluation
- **Dataset:** The dataset is preprocessed, categorical variables are encoded, and numerical values are normalized.
- **Models trained:** Linear Regression, Random Forest, Gradient Boosting, and SVR.
- **Best performing model:** The **Random Forest Regressor** (saved in `models/` folder).
- **Metrics used:** Mean Squared Error (MSE), Mean Absolute Error (MAE), R-Squared (R²).

## Future Improvements
- Deploying the model to **AWS/GCP/Azure**.
- Improving the **UI/UX** of the frontend.
- Adding **Docker support** for easier deployment.

## License
This project is licensed under the **MIT License**.

---
**Contributors:** Your Name (your-email@example.com)

