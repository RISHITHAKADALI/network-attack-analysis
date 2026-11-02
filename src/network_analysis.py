
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc


INPUT_PATH = "outputs/cleaned_data.csv"
OUTPUT_PATH = "outputs/network_connections"


def main():
    spark = (
        SparkSession.builder
        .appName("NetworkAttackAnalysis")
        .master("local[2]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        print("\n=== LARGE-SCALE NETWORK ANALYSIS ===")

        # Load the cleaned dataset
        df = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(INPUT_PATH)
        )

        # Remove whitespace from column names
        df = df.toDF(*[name.strip() for name in df.columns])

        print("Total traffic records:", df.count())

        # Analyze communication between IP addresses
        connections = (
            df.groupBy("Source IP", "Destination IP")
            .agg(count("*").alias("Connection Count"))
            .orderBy(desc("Connection Count"))
        )

        print("\n=== TOP 10 NETWORK CONNECTIONS ===")
        connections.show(10, truncate=False)

        # Display the most frequent source IP addresses
        print("\n=== TOP 10 SOURCE IP ADDRESSES ===")

        df.groupBy("Source IP") \
            .agg(count("*").alias("Traffic Count")) \
            .orderBy(desc("Traffic Count")) \
            .show(10, truncate=False)

        # Display the most frequent destination IP addresses
        print("\n=== TOP 10 DESTINATION IP ADDRESSES ===")

        df.groupBy("Destination IP") \
            .agg(count("*").alias("Traffic Count")) \
            .orderBy(desc("Traffic Count")) \
            .show(10, truncate=False)

        # Save the connection analysis
        # Use Pandas to avoid the current Windows Hadoop write issue
        import os
        os.makedirs("outputs", exist_ok=True)

        connections.limit(1000).toPandas().to_csv(
            "outputs/top_network_connections.csv",
            index=False
        )

        print("\nTop 1000 connection records saved to:")
        print("outputs/top_network_connections.csv")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
