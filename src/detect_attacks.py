
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc, sum as spark_sum


INPUT_PATH = "outputs/cleaned_data.csv"
OUTPUT_PATH = "outputs/ddos_detection_results.csv"


def main():
    spark = (
        SparkSession.builder
        .appName("DDoSAttackDetection")
        .master("local[2]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        print("\n=== DDoS ATTACK DETECTION ===")

        # Load the cleaned dataset
        df = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(INPUT_PATH)
        )

        # Normalize column names
        df = df.toDF(*[name.strip() for name in df.columns])

        # Count records for each label
        print("\n=== TRAFFIC LABEL SUMMARY ===")

        label_counts = (
            df.groupBy("Label")
            .agg(count("*").alias("Record Count"))
            .orderBy(desc("Record Count"))
        )

        label_counts.show(truncate=False)

        # Calculate percentage of each label
        total_records = df.count()

        summary = (
            label_counts
            .withColumn(
                "Percentage",
                col("Record Count") * 100.0 / total_records
            )
        )

        print("\n=== TRAFFIC DISTRIBUTION (%) ===")
        summary.show(truncate=False)

        # Filter records already labelled as DDoS
        ddos_df = df.filter(
            col("Label").isin("DDoS", "DoS Hulk", "DoS GoldenEye",
                              "DoS slowloris", "DoS Slowhttptest")
        )

        print("\n=== DDoS-RELATED RECORDS ===")
        print("DDoS-related records:", ddos_df.count())

        # Find source-destination pairs with the most DDoS records
        suspicious_pairs = (
            ddos_df
            .groupBy("Source IP", "Destination IP")
            .agg(count("*").alias("DDoS Record Count"))
            .orderBy(desc("DDoS Record Count"))
        )

        print("\n=== TOP 10 SOURCE-DESTINATION PAIRS ===")
        suspicious_pairs.show(10, truncate=False)

        # Save results to a single CSV file
        os.makedirs("outputs", exist_ok=True)

        summary.toPandas().to_csv(
            "outputs/traffic_label_summary.csv",
            index=False
        )

        suspicious_pairs.limit(1000).toPandas().to_csv(
            OUTPUT_PATH,
            index=False
        )

        print("\nResults saved successfully:")
        print("outputs/traffic_label_summary.csv")
        print(OUTPUT_PATH)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
