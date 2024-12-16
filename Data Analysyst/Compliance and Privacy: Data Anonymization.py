import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from hashlib import sha256

# Load the dataset
df = pd.read_csv('customer_data.csv')

# Remove Direct Identifiers
direct_identifiers = ['full_name', 'email', 'phone_number']
df = df.drop(columns=direct_identifiers)

# Hash Indirect Identifiers (e.g., customer_id)
df['customer_id'] = df['customer_id'].apply(lambda x: sha256(str(x).encode('utf-8')).hexdigest())

# K-Anonymity: Generalize sensitive numeric data
# Here we'll bin age into age groups for k-anonymity
k = 5  # Example k value for k-anonymity
age_binner = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
df['age_group'] = age_binner.fit_transform(df[['age']]).astype(int)
df = df.drop('age', axis=1)

# Generalize other sensitive data like 'income' into bands
income_binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
df['income_band'] = income_binner.fit_transform(df[['income']]).astype(int)
df = df.drop('income', axis=1)

# Differential Privacy: Add noise to numeric features (monetary data)
epsilon = 1.0  # Privacy budget
sensitivity = df['total_spend'].max() - df['total_spend'].min()  # data range
scale = sensitivity / epsilon
noisy_total_spend = df['total_spend'] + np.random.laplace(0, scale, df.shape[0])
df['noisy_total_spend'] = noisy_total_spend.clip(lower=0)  # ensure non-negative values

# Check for uniqueness to ensure k-anonymity
grouped = df.groupby(['age_group', 'income_band', 'zip_code']).size()
if any(grouped < k):
    print("Warning: Some groups do not meet k-anonymity threshold.")

# Save anonymized data
df.to_csv('anonymized_customer_data.csv', index=False)

# Verify anonymization (example, check for unique records)
print("Number of unique records in original data:", df_original.shape[0])
print("Number of unique records after anonymization:", df.shape[0])
