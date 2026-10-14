# Import libraries
from tkinter import _test

import joblib # type: ignore
import os


from sklearn.metrics import classification_report # type: ignore
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score


from src.preprocessing import load_data
from src.feature_engineering import engineer_features
from src.train import train_model


def evaluate_model():

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    # Load model
    model_path = os.path.join(BASE_DIR, "models", "tourism_model.pkl")

    model = joblib.load(model_path)

    # Load dataset
    file_path = os.path.join(BASE_DIR, "data", "cleaned_tourism_data.csv")

    df = load_data(file_path)

    # Feature engineering
    df = engineer_features(df)

    target = "fraud_flag_or_label"   # must match train.py

    X = df.drop(columns=[target])
    y = df[target]

    # Predict
    y_pred = model.predict(X)

    # Evaluation
    print(classification_report(y, y_pred))

    # Model cross evalutions:
    # 
    cm = confusion_matrix(y,y_pred)  # confusion matrix
    print("\n---------------------------")
    print("confusion matrix")
    print("-----------------------------\n")

    print(cm)

    scores = cross_val_score(model, X, y, cv=5)  # cross-validation 
    print("\n---------------------------------")
    print(" Cross validation")
    print("------------------------------------")

    print("scores :", scores)

    print("Average: ", scores.mean())

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:,1]

        roc_auc = roc_auc_score(y, y_prob)     # roc_auc_score
        print("\n------------------------")
        print(" Roc-AUC Score")
        print("---------------------------\n")
        
        print(round(roc_auc))



    


    


if __name__ == "__main__":
    evaluate_model()