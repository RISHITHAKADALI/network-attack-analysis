# Large-Scale Network Analysis for Attack Pattern Detection

A data-analysis prototype that cleans network-flow records, summarizes source-to-destination connections, visualizes traffic labels, and trains a baseline machine-learning classifier to separate **BENIGN** and **DDoS** flows. Results are shown in an interactive Streamlit dashboard.

> Student project. It analyzes an existing labelled dataset. It does not capture or monitor live network traffic.

## Features

- Load and validate a network-flow CSV dataset
- Clean the data (whitespace, infinite values, duplicates)
- Count connections per Source IP and Destination IP pair
- Summarize traffic labels and find the busiest DDoS source/destination pairs
- Generate charts (label distribution, top connections, top DDoS connections)
- Train and evaluate a Logistic Regression classifier
- Explore everything in a four-page Streamlit dashboard

## Dataset

- Network-flow CSV with 225,745 records and 80 numeric features
- Two classes: DDoS (128,027 records, 56.71%) and BENIGN (97,718 records, 43.29%)
- Appears to be the CIC-IDS2017 Friday-afternoon DDoS capture (confirm and add source link)

The dataset is not stored in this repository because of its size. Download it and place it at:

```
data/sample.csv
```

## Project Structure

```
.
├── app.py                     # Streamlit dashboard
├── requirements.txt
├── README.md
├── data/
│   └── sample.csv             # dataset (not included in the repo)
├── outputs/                   # generated CSVs, charts, predictions
└── src/
    ├── load_data.py           # validate the dataset, print columns and row count
    ├── preprocess.py          # clean the data -> outputs/cleaned_data.csv
    ├── network_analysis.py    # connection counts per IP pair
    ├── detect_attacks.py      # label summary and DDoS pair analysis
    ├── visualize.py           # PNG charts
    └── train_model.py         # train and evaluate the classifier
```

## Tools and Technologies

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit, Joblib (PySpark is listed in requirements for scaling to larger data).

## Setup

1. Clone the repository:

   ```
   git clone https://github.com/RISHITHAKADALI/network-attack-analysis.git
   cd network-attack-analysis
   ```

2. Create and activate a virtual environment (Windows PowerShell):

   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Put the dataset at `data/sample.csv`.

## How to Run

Run the scripts in this order from the project folder:

```
python src/load_data.py
python src/preprocess.py
python src/network_analysis.py
python src/detect_attacks.py
python src/visualize.py
python src/train_model.py
```

Then start the dashboard:

```
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Dashboard Pages

| Page | What it shows |
|---|---|
| Overview | Total records, DDoS count and share, label distribution and charts |
| Network Connections | Top source-to-destination connections with an adjustable table and chart |
| Attack Analysis | Label summary and top DDoS-labelled source/destination pairs |
| Model Evaluation | Test accuracy, classification report, confusion matrix |

## Results

| Item | Value |
|---|---|
| Total records | 225,745 |
| DDoS-labelled records | 128,027 (56.71%) |
| Training / test records | 180,596 / 45,149 |
| Test accuracy | 99.87% |
| Confusion matrix (BENIGN, DDoS) | [[19512, 32], [25, 25580]] |

Key findings:

- 128,024 of the 128,027 DDoS-labelled records go from `172.16.0.1` to `192.168.10.50`, so one source floods one server.
- On the test set, 32 BENIGN flows were flagged as DDoS and 25 DDoS flows were missed.

## Methodology

1. Clean the data and remove duplicate rows.
2. Keep only BENIGN and DDoS rows for training.
3. Drop identifier columns (Flow ID, IPs, Timestamp) and keep numeric features.
4. Split 80/20 with stratification.
5. Pipeline: median imputer, StandardScaler, Logistic Regression (balanced class weights).
6. Evaluate with accuracy, classification report and confusion matrix.

## Limitations

- Uses existing dataset labels. It does not detect live attacks.
- A random train/test split can overestimate performance when near-duplicate flows exist, so the accuracy is a preliminary result.
- Only one attack type (DDoS) and one simple model are used.

## Future Work

- Add other attack types such as PortScan and brute force
- Compare Random Forest and XGBoost
- Use time-based train/test splits
- Use PySpark for much larger datasets
- Add real-time packet capture

## Author

Rishitha Kadali

## References

- Sharafaldin, I., Lashkari, A. H., Ghorbani, A. A. "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization." ICISSP, 2018 (CIC-IDS2017).
- Pedregosa, F. et al. "Scikit-learn: Machine Learning in Python." Journal of Machine Learning Research, 12, 2011.
- Streamlit documentation: https://docs.streamlit.io
