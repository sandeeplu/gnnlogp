import pandas as pd

import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



# File paths

test_predictions_path = "test_predictions.csv"

data_path = "/home/a4724/chemprop/my_pred/data.csv"

output_path = "merged_predictions.csv"



# Read CSV files

test_df = pd.read_csv(test_predictions_path)   # contains smiles, pred_exp

data_df = pd.read_csv(data_path)               # contains smiles, exp



# Merge on smiles

df = pd.merge(test_df, data_df[['smiles', 'exp']], on='smiles', how='left')



# Save merged file

df.to_csv(output_path, index=False)

print(f"Merged file saved as: {output_path}")



# Extract true and predicted values

y_true = df['exp'].values

y_pred = df['pred_exp'].values



# Compute metrics

mse = mean_squared_error(y_true, y_pred)

rmse = np.sqrt(mse)

mae = mean_absolute_error(y_true, y_pred)

r2 = r2_score(y_true, y_pred)



# Print results

print(f"MSE: {mse:.4f}")

print(f"RMSE: {rmse:.4f}")

print(f"MAE: {mae:.4f}")

print(f"R²: {r2:.4f}")


