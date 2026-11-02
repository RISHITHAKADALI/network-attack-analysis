
import os
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Network Attack Detection",
    page_icon="🛡️",
    layout="wide",
)

OUTPUT_DIR = "outputs"

st.title("🛡️ Large-Scale Network Analysis")
st.subheader("Network Traffic and Attack Pattern Detection")
st.caption(
    "CIC-IDS-2017 Dataset | BENIGN vs DDoS Analysis"
)

# --------------------------------------------------
# 2. HELPER FUNCTION TO LOAD CSV FILES
# --------------------------------------------------

@st.cache_data
def load_csv(filename):
    path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(path):
        return None

    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as error:
        st.error(f"Could not read {filename}: {error}")
        return None


# --------------------------------------------------
# 3. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🛡️ Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Overview",
        "Network Connections",
        "Attack Analysis",
        "Model Evaluation",
    ],
)

# --------------------------------------------------
# 4. OVERVIEW PAGE
# --------------------------------------------------

if page == "Overview":

    st.header("Project Overview")

    summary = load_csv("traffic_label_summary.csv")

    if summary is None:
        st.warning(
            "Traffic summary not found. Run "
            "python src\\detect_attacks.py first."
        )
    else:
        summary.columns = summary.columns.str.strip()

        if {"Label", "Record Count"}.issubset(summary.columns):

            summary["Record Count"] = pd.to_numeric(
                summary["Record Count"],
                errors="coerce",
            ).fillna(0)

            total_records = int(summary["Record Count"].sum())

            benign_records = int(
                summary.loc[
                    summary["Label"].astype(str).str.upper()
                    == "BENIGN",
                    "Record Count",
                ].sum()
            )

            ddos_records = int(
                summary.loc[
                    summary["Label"].astype(str).str.upper()
                    == "DDOS",
                    "Record Count",
                ].sum()
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Traffic Records",
                f"{total_records:,}",
            )

            col2.metric(
                "BENIGN Records",
                f"{benign_records:,}",
            )

            col3.metric(
                "DDoS-Labelled Records",
                f"{ddos_records:,}",
            )

            st.divider()

            st.subheader("Traffic Distribution")

            st.bar_chart(
                summary.set_index("Label")["Record Count"],
                width="stretch",
            )

            st.subheader("Traffic Summary Table")

            st.dataframe(
                summary,
                width="stretch",
                hide_index=True,
            )

        else:
            st.error(
                "Expected columns 'Label' and 'Record Count' "
                "were not found in the summary file."
            )

    st.divider()

    st.header("Generated Charts")

    chart_files = [
        (
            "Traffic Distribution",
            "traffic_distribution.png",
        ),
        (
            "Top Network Connections",
            "top_connections.png",
        ),
        (
            "DDoS-Labelled Connections",
            "ddos_connections.png",
        ),
    ]

    for title, filename in chart_files:
        path = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(path):
            st.subheader(title)
            st.image(path, width="stretch")

# --------------------------------------------------
# 5. NETWORK CONNECTIONS PAGE
# --------------------------------------------------

elif page == "Network Connections":

    st.header("Network Connection Analysis")

    connections = load_csv("top_network_connections.csv")

    if connections is None:
        st.warning(
            "Network connections file not found. "
            "Run python src\\network_analysis.py first."
        )
    else:
        connections.columns = connections.columns.str.strip()

        st.subheader("Top Network Connections")

        if not connections.empty:

            max_rows = min(100, len(connections))

            number = st.slider(
                "Number of connections to display",
                min_value=1,
                max_value=max_rows,
                value=min(10, max_rows),
            )

            st.dataframe(
                connections.head(number),
                width="stretch",
                hide_index=True,
            )

            if "Connection Count" in connections.columns:
                st.subheader("Connection Frequency")

                chart_data = connections.head(number).copy()

                if {
                    "Source IP",
                    "Destination IP",
                    "Connection Count",
                }.issubset(chart_data.columns):

                    chart_data["Connection"] = (
                        chart_data["Source IP"].astype(str)
                        + " → "
                        + chart_data["Destination IP"].astype(str)
                    )

                    chart_data = chart_data.set_index("Connection")

                    st.bar_chart(
                        chart_data["Connection Count"],
                        horizontal=True,
                        width="stretch",
                    )

        else:
            st.info("The connections file contains no records.")

# --------------------------------------------------
# 6. ATTACK ANALYSIS PAGE
# --------------------------------------------------

elif page == "Attack Analysis":

    st.header("Attack Pattern Analysis")

    ddos = load_csv("ddos_detection_results.csv")

    if ddos is None:
        st.warning(
            "Attack analysis results not found. "
            "Run python src\\detect_attacks.py first."
        )
    else:
        ddos.columns = ddos.columns.str.strip()

        st.subheader("DDoS-Labelled Network Connections")

        if ddos.empty:
            st.info("No matching attack-labelled records were found.")
        else:
            st.dataframe(
                ddos.head(100),
                width="stretch",
                hide_index=True,
            )

            if {
                "Source IP",
                "Destination IP",
                "DDoS Record Count",
            }.issubset(ddos.columns):

                ddos["DDoS Record Count"] = pd.to_numeric(
                    ddos["DDoS Record Count"],
                    errors="coerce",
                ).fillna(0)

                ddos["Connection"] = (
                    ddos["Source IP"].astype(str)
                    + " → "
                    + ddos["Destination IP"].astype(str)
                )

                st.subheader("DDoS-Labelled Record Counts")

                chart_data = (
                    ddos.head(10)
                    .set_index("Connection")["DDoS Record Count"]
                )

                st.bar_chart(
                    chart_data,
                    horizontal=True,
                    width="stretch",
                )

    st.info(
        "This page summarizes labels in the supplied dataset. "
        "It does not represent live network monitoring."
    )

# --------------------------------------------------
# 7. MACHINE LEARNING EVALUATION PAGE
# --------------------------------------------------

elif page == "Model Evaluation":

    st.header("Machine Learning Model Evaluation")

    predictions = load_csv("model_test_predictions.csv")

    if predictions is None:
        st.warning(
            "Test predictions were not found. Run "
            "python src\\train_model.py first."
        )

    elif {
        "Actual Label",
        "Predicted Label",
    }.issubset(predictions.columns):

        actual = predictions["Actual Label"]
        predicted = predictions["Predicted Label"]

        accuracy = accuracy_score(actual, predicted)

        st.metric(
            "Test Accuracy",
            f"{accuracy:.2%}",
        )

        st.caption(
            "This score is from a random held-out test split of "
            "the existing dataset. It does not guarantee equivalent "
            "performance on live or independent network traffic."
        )

        st.divider()

        st.subheader("Classification Report")

        report = classification_report(
            actual,
            predicted,
            labels=["BENIGN", "DDoS"],
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4),
            width="stretch",
        )

        st.divider()

        st.subheader("Confusion Matrix")

        matrix = confusion_matrix(
            actual,
            predicted,
            labels=["BENIGN", "DDoS"],
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=["Actual BENIGN", "Actual DDoS"],
            columns=["Predicted BENIGN", "Predicted DDoS"],
        )

        st.dataframe(
            matrix_df,
            width="stretch",
        )

        st.subheader("Actual vs Predicted Labels")

        comparison = pd.DataFrame({
            "Actual": actual.value_counts(),
            "Predicted": predicted.value_counts(),
        }).fillna(0)

        st.bar_chart(
            comparison,
            width="stretch",
        )

    else:
        st.error(
            "The prediction file must contain 'Actual Label' "
            "and 'Predicted Label' columns."
        )

# --------------------------------------------------
# 8. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Academic project: Large-Scale Network Analysis for "
    "Attack Pattern Detection | Dataset-based analysis, "
    "not a live intrusion detection system."
)
