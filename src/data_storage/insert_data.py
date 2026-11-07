import pandas as pd
from pathlib import Path
from mysql_connection import engine

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset path
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step7_feature_engineered.csv"
)

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)

# Insert data into MySQL
df.to_sql(
    name="job_acceptance_data",
    con=engine,
    if_exists="replace",
    index=False,
    chunksize=1000,
    method="multi"
)

print("Data inserted successfully into MySQL!")