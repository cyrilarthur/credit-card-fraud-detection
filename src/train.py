# Import libraries
import joblib # type: ignore
import os

from sklearn.pipeline import Pipeline # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

from src.preprocessing import load_data, create_preprocessor
from src.feature_engineering import engineer_features


def train_model():

    # Get absolute dataset path (prevents errors)
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    file_path = os.path.join(
        BASE_DIR,
        "data",
        "cleaned_tourism_data.csv"
    )

    # Load data
    df = load_data(file_path)

    # Feature engineering
    df = engineer_features(df)

    # -----------------------------
    # TARGET COLUMN (IMPORTANT)
    # Change this based on your dataset
    # -----------------------------
    target = "fraud_flag_or_label"  

    # Split features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Create preprocessing pipeline
    preprocessor = create_preprocessor(df)

    # Build ML pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    pipeline.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(BASE_DIR, "models", "tourism_model.pkl")  

    joblib.dump(pipeline, model_path)

    print("Model trained and saved successfully!")

    return pipeline, X_test, y_test


if __name__ == "__main__":
    train_model()