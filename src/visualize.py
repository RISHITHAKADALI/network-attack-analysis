
import os
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Chart 1: Traffic distribution
    summary = pd.read_csv(
        os.path.join(OUTPUT_DIR, "traffic_label_summary.csv")
    )

    plt.figure(figsize=(8, 5))
    plt.bar(summary["Label"], summary["Record Count"])
    plt.title("Network Traffic by Label")
    plt.xlabel("Traffic Label")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "traffic_distribution.png"), dpi=300)
    plt.close()

    # Chart 2: Top 10 network connections
    connections = pd.read_csv(
        os.path.join(OUTPUT_DIR, "top_network_connections.csv")
    ).head(10)

    connections["Connection"] = (
        connections["Source IP"].astype(str)
        + " -> "
        + connections["Destination IP"].astype(str)
    )

    plt.figure(figsize=(12, 6))
    plt.barh(connections["Connection"], connections["Connection Count"])
    plt.title("Top 10 Network Connections")
    plt.xlabel("Number of Records")
    plt.ylabel("Source -> Destination")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_connections.png"), dpi=300)
    plt.close()

    # Chart 3: DDoS-labelled connections
    ddos = pd.read_csv(
        os.path.join(OUTPUT_DIR, "ddos_detection_results.csv")
    ).head(10)

    ddos["Connection"] = (
        ddos["Source IP"].astype(str)
        + " -> "
        + ddos["Destination IP"].astype(str)
    )

    plt.figure(figsize=(12, 6))
    plt.barh(ddos["Connection"], ddos["DDoS Record Count"])
    plt.title("Top DDoS-Labelled Network Connections")
    plt.xlabel("DDoS-Labelled Records")
    plt.ylabel("Source -> Destination")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ddos_connections.png"), dpi=300)
    plt.close()

    print("\nAll charts generated successfully!")
    print("Check the outputs folder for:")
    print("- traffic_distribution.png")
    print("- top_connections.png")
    print("- ddos_connections.png")


if __name__ == "__main__":
    main()
