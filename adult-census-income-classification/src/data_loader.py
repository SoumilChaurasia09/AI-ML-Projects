import os
import urllib.request
import pandas as pd
import numpy as np

# Official UCI Adult Dataset URLs
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

def download_data(dest_dir="data"):
    """Downloads adult.data and adult.test if they do not exist."""
    os.makedirs(dest_dir, exist_ok=True)
    
    train_path = os.path.join(dest_dir, "adult.data")
    test_path = os.path.join(dest_dir, "adult.test")
    
    if not os.path.exists(train_path):
        print(f"Downloading training data from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, train_path)
        print("Training data downloaded.")
    else:
        print("Training data already exists locally.")
        
    if not os.path.exists(test_path):
        print(f"Downloading test data from {TEST_URL}...")
        urllib.request.urlretrieve(TEST_URL, test_path)
        print("Test data downloaded.")
    else:
        print("Test data already exists locally.")
        
    return train_path, test_path

def load_and_clean_data(dest_dir="data"):
    """
    Downloads, loads and performs initial cleaning of the Adult Census Income dataset.
    Returns:
        train_df, test_df
    """
    train_path, test_path = download_data(dest_dir)
    
    # Read training data
    # Note: adult.data has no header row
    train_df = pd.read_csv(train_path, header=None, names=COLUMNS, skipinitialspace=True)
    
    # Read test data
    # Note: adult.test has a dummy first line (e.g. '|1x3 Cross validator') which we must skip
    test_df = pd.read_csv(test_path, header=None, names=COLUMNS, skipinitialspace=True, skiprows=1)
    
    # 1. Clean string columns (strip extra spaces if skipinitialspace didn't cover it fully)
    for col in train_df.select_dtypes(include=['object']).columns:
        train_df[col] = train_df[col].astype(str).str.strip()
        test_df[col] = test_df[col].astype(str).str.strip()
        
    # 2. Replace '?' with NaN
    train_df = train_df.replace('?', np.nan)
    test_df = test_df.replace('?', np.nan)
    
    # 3. Clean target variable 'income'
    # Test set targets have a trailing period: '<=50K.' and '>50K.'
    test_df['income'] = test_df['income'].str.rstrip('.')
    
    # Check target consistency
    valid_targets = {'<=50K', '>50K'}
    train_df = train_df[train_df['income'].isin(valid_targets)].copy()
    test_df = test_df[test_df['income'].isin(valid_targets)].copy()
    
    # Convert target to binary: <=50K -> 0, >50K -> 1
    train_df['income_bin'] = (train_df['income'] == '>50K').astype(int)
    test_df['income_bin'] = (test_df['income'] == '>50K').astype(int)
    
    return train_df, test_df

if __name__ == "__main__":
    print("Testing data loader...")
    train, test = load_and_clean_data("data")
    print(f"Train set shape: {train.shape}")
    print(f"Test set shape: {test.shape}")
    print("Missing values in Train set:\n", train.isnull().sum()[train.isnull().sum() > 0])
    print("Missing values in Test set:\n", test.isnull().sum()[test.isnull().sum() > 0])
    print("Data loader script works successfully!")
