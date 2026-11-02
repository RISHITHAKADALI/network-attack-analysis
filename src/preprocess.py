
import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim


# File paths
INPUT_PATH = "data/sample.csv"
OUTPUT_PATH = "outputs/cleaned_data.csv"


def main():
    spark = (
        SparkSession.builder
        .appName("NetworkAttackPreprocessing")
        .master("local[2]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        # Step 1: Check whether the dataset exists
        if not Path(INPUT_PATH).exists():
            raise FileNotFoundError(
                f"Dataset not found: {INPUT_PATH}\n"
                "Make sure sample.csv is inside the data folder."
            )

        print("\n=== NETWORK ATTACK DATA PREPROCESSING ===")

        # Step 2: Load the CSV dataset using Spark
        df = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(INPUT_PATH)
        )

        # Step 3: Remove extra spaces from column names
        df = df.toDF(*[name.strip() for name in df.columns])

        original_count = df.count()
        print(f"\nOriginal row count: {original_count}")

        # Step 4: Check required columns
        required_columns = [
            "Source IP",
            "Destination IP",
            "Label",
        ]

        missing_columns = [
            name for name in required_columns
            if name not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Required columns are missing: {missing_columns}"
            )

        # Step 5: Remove rows with null values
        df = df.filter(
            col("Source IP").isNotNull()
            & col("Destination IP").isNotNull()
            & col("Label").isNotNull()
        )

        # Step 6: Remove rows with empty required fields
        df = df.filter(
            (trim(col("Source IP")) != "")
            & (trim(col("Destination IP")) != "")
            & (trim(col("Label")) != "")
        )

        cleaned_count = df.count()
        print(f"Cleaned row count: {cleaned_count}")
        print(f"Rows removed: {original_count - cleaned_count}")

        # Step 7: Display attack label counts
        print("\n=== ATTACK LABEL COUNTS ===")

        label_counts = (
            df.groupBy("Label")
            .count()
            .orderBy("Label")
        )

        label_counts.show(truncate=False)

        # Step 8: Create the output directory
        os.makedirs("outputs", exist_ok=True)

        # Step 9: Save the cleaned data as one CSV file.
        # This uses Pandas for local file output to avoid
        # Spark's Windows Hadoop winutils.exe write issue.
        cleaned_pandas = df.toPandas()

        cleaned_pandas.to_csv(
            OUTPUT_PATH,
            index=False,
            encoding="utf-8",
        )

        print(f"\nCleaned data saved to: {OUTPUT_PATH}")
        print("\nPreprocessing completed successfully!")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
