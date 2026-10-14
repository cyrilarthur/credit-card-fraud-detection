# Import pandas
import pandas as pd # type: ignore


# Import sklearn tools
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.preprocessing import ( # type: ignore
    OneHotEncoder,
    StandardScaler
)


# Load CSV file
def load_data(file_path):

    df = pd.read_csv(file_path)

    return df



# Create preprocessing pipeline
def create_preprocessor(df):

    # Target column
    target = "fraud_flag_or_label"


    # Remove target
    X = df.drop(
        columns=[target]
    )


    # Numeric columns
    numeric_features = (
        X.select_dtypes(
            include=[
                "int64",
                "float64"
            ]
        )
        .columns
        .tolist()
    )


    # Categorical columns
    categorical_features = (
        X.select_dtypes(
            include=[
                "object"
            ]
        )
        .columns
        .tolist()
    )


    # Build transformer
    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numeric",
                StandardScaler(),
                numeric_features
            ),


            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )

        ]

    )


    return preprocessor