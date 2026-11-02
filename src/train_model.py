
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# File paths
DATA_PATH = "outputs/cleaned_data.csv"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "ddos_model.joblib")
PREDICTIONS_PATH = os.path.join(
    OUTPUT_DIR, "model_test_predictions.csv"
)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 55)
    print(" NETWORK ATTACK DETECTION - MACHINE LEARNING")
    print("=" * 55)

    # 1. Load the dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    print("\nLoading cleaned network traffic data...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError("The dataset does not contain a Label column.")

    # 2. Keep BENIGN and DDoS records
    df = df[df["Label"].isin(["BENIGN", "DDoS"])].copy()

    print(f"Total records: {len(df)}")
    print("\nClass counts:")
    print(df["Label"].value_counts())

    # 3. Select numeric traffic features.
    # Exclude labels, IP addresses, timestamps and identifiers.
    excluded_columns = {
        "Label",
        "Source IP",
        "Destination IP",
        "Timestamp",
        "Flow ID",
    }

    feature_columns = [
        column for column in df.columns
        if column not in excluded_columns
    ]

    X = df[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # Replace infinite values with missing values
    X = X.replace([float("inf"), float("-inf")], float("nan"))

    # Remove columns with no valid numeric values
    X = X.dropna(axis=1, how="all")

    # Keep rows containing at least one usable feature
    usable_rows = X.notna().any(axis=1)
    X = X.loc[usable_rows]
    y = df.loc[X.index, "Label"]

    if X.empty or X.shape[1] == 0:
        raise ValueError("No usable numeric features were found.")

    if y.nunique() != 2:
        raise ValueError(
            "Both BENIGN and DDoS classes are required."
        )

    # 4. Split data into training and testing sets
    # The test set is held out until evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"\nTraining records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Numeric features: {X.shape[1]}")

    # 5. Create the machine learning pipeline
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    )

    # 6. Train the model
    print("\nTraining the model...")
    model.fit(X_train, y_train)
    print("Training completed!")

    # 7. Evaluate on the held-out test set
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 55)
    print(" MODEL EVALUATION")
    print("=" * 55)

    print(f"\nTest accuracy: {accuracy:.4%}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["BENIGN", "DDoS"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=true, columns=predicted):")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=["BENIGN", "DDoS"],
        )
    )

    # 8. Save test predictions
    results = pd.DataFrame({
        "Actual Label": y_test,
        "Predicted Label": predictions,
    }, index=y_test.index)

    results.to_csv(
        PREDICTIONS_PATH,
        index_label="Original Row Index",
    )
    print(f"\nTest predictions saved to: {PREDICTIONS_PATH}")

    # 9. Save the trained model and its feature names
    model_package = {
        "model": model,
        "feature_columns": list(X.columns),
        "classes": list(model.classes_),
    }

    joblib.dump(model_package, MODEL_PATH)

    print(f"Trained model saved to: {MODEL_PATH}")

    print("\n" + "=" * 55)
    print(" PROJECT MODEL TRAINING COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    main()
