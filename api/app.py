# Import FastAPI framework
from fastapi import FastAPI # type: ignore

# Import Pydantic for request validation
from pydantic import BaseModel # type: ignore

# Import pandas
import pandas as pd # type: ignore

# Import joblib to load saved model
import joblib # type: ignore

# Import os for file paths
import os

# Import feature engineering
from src.feature_engineering import engineer_features


# Create FastAPI application
app = FastAPI(
    title="Credit Card Fraud Detection API"
)


# Get project root directory
BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)


# Build model path
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tourism_model.pkl"
)


# Load trained model
model = joblib.load(
    MODEL_PATH
)


# Request data structure
class Transaction(BaseModel):

    transaction_date_and_time: str

    transaction_amount: float

    cardholder_name: str

    card_number: str

    merchant_name: str

    merchant_category_code: int

    transaction_location: str

    transaction_currency: str

    card_type: str

    card_expiration_date: str

    cvv_code: str

    transaction_response_code: int

    transaction_id: str

    previous_transactions: str

    transaction_source: str

    ip_address: str

    device_information: str

    user_account_information: str

    transaction_notes: str



# Home endpoint
@app.get("/")
def home():

    return {
        "message": "Fraud Detection API Running"
    }



# Prediction endpoint
@app.post("/predict")
def predict(transaction: Transaction):

    # Convert request object into dictionary
    data = transaction.model_dump()


    # Convert dictionary into dataframe
    df = pd.DataFrame(
        [data]
    )


    # Rename columns to match training dataset
    df = df.rename(
        columns={
            "card_number":
            "card_number_(hashed_or_encrypted)",

            "merchant_category_code":
            "merchant_category_code_(mcc)",

            "transaction_location":
            "transaction_location_(city_or_zip_code)",

            "cvv_code":
            "cvv_code_(hashed_or_encrypted)"
        }
    )


    # Apply same feature engineering used during training
    df = engineer_features(df)


    # Make prediction
    prediction = model.predict(df)[0]


    # Return prediction
    return {
        "fraud_prediction": int(prediction)
    }