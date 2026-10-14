# Import numpy for mathematical operations
import numpy as np # type: ignore


# Function for creating new fraud detection features
def engineer_features(df):

    # Create high transaction amount feature
    if "transaction_amount" in df.columns:

        df["is_high_amount"] = (
            df["transaction_amount"]
            >
            df["transaction_amount"].mean()
        ).astype(int)


        # Apply logarithmic transformation
        # Helps reduce impact of extremely large transactions
        df["log_transaction_amount"] = np.log1p(
            df["transaction_amount"]
        )


    # Create foreign transaction feature
    if "transaction_currency" in df.columns:

        df["is_foreign_transaction"] = (
            df["transaction_currency"] != "USD"
        ).astype(int)


    # Count previous transactions per user
    if "user_account_information" in df.columns:

        df["user_transaction_count"] = (
            df.groupby(
                "user_account_information"
            )["transaction_id"]
            .transform("count")
        )


    # Fill missing values
    df = df.fillna(0)


    # Return processed dataframe
    return df