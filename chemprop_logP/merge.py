import pandas as pd

# Load original data with experimental values
orig_data = pd.read_csv('data.csv')  # CSV with columns CMPD_CHEMBLID, exp, smiles

# Load predictions from chemprop
preds = pd.read_csv('predictions.csv')  # columns: CMPD_CHEMBLID, predicted_value, smiles

# Merge on compound ID or SMILES
merged = orig_data.merge(preds, on=['CMPD_CHEMBLID', 'smiles'])

print(merged[['CMPD_CHEMBLID', 'exp', 'pred_exp']])

# Save merged dataframe to CSV

merged.to_csv('merged_results.csv', index=False)

print("✅ Merged file saved as 'merged_results.csv'")
