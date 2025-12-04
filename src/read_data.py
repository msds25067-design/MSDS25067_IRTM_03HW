import pandas as pd
import os

# Path to dataset
DATA_DIR = "../data"
DATA_FILE = "articles.csv"  # replace with your actual filename
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

# Try reading with a different encoding
try:
    df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')  # alternative encodings: 'cp1252', 'latin1'
    print(f"Dataset loaded successfully! Shape: {df.shape}")
except FileNotFoundError:
    print(f"File not found at path: {DATA_PATH}")
    exit()
except UnicodeDecodeError as e:
    print(f"Encoding error: {e}")
    exit()

# Show basic info
print("\nColumns in dataset:")
print(df.columns)

