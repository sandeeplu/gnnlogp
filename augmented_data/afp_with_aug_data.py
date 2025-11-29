# ==========================================================
# AttentiveFP for Lipophilicity Prediction (Full Dataset)
# ==========================================================

import os
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import sqrt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP
from torch_geometric.utils import to_networkx, from_smiles
from rdkit import Chem

# -------------------- SETUP -------------------------------
start_time = time.time()
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def seed_set(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_set(42)
torch.use_deterministic_algorithms(True)
generator = torch.Generator().manual_seed(42)

# -------------------- DATA LOADING -------------------------
df_final = pd.read_csv('../Lipophilicity_aug.csv')
#df_final = pd.read_csv('full_dataset_cleaned.csv')
print(f"✅ Loaded dataset: {len(df_final)} molecules")
print(df_final.head())

# -------------------- GRAPH CREATION -----------------------
graph_list = []
for i, smile in enumerate(df_final['smiles']):
    mol_graph = from_smiles(smile)
    mol_graph.x = mol_graph.x.float()
    mol_graph.y = torch.tensor([df_final['exp'][i]], dtype=torch.float)
    graph_list.append(mol_graph)

# -------------------- DATA SPLIT ---------------------------
random.shuffle(graph_list)
train_size = int(0.8 * len(graph_list))
test_size = len(graph_list) - train_size
train_dataset, test_dataset = random_split(graph_list, [train_size, test_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=generator)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -------------------- MODEL SETUP --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttentiveFP(
    in_channels=9, hidden_channels=192, out_channels=1,
    edge_dim=3, num_layers=6, num_timesteps=2, dropout=0.1
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=4.41e-4, weight_decay=4.86e-5)
torch.cuda.empty_cache()

# -------------------- TRAIN & TEST FUNCTIONS ---------------
def train(loader):
    model.train()
    total_loss = total_mae = total_samples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        mse_loss = F.mse_loss(out, data.y.view(-1, 1))
        mae_loss = F.l1_loss(out, data.y.view(-1, 1))
        mse_loss.backward()
        optimizer.step()
        total_loss += mse_loss.item() * data.num_graphs
        total_mae += mae_loss.item() * data.num_graphs
        total_samples += data.num_graphs
    return np.sqrt(total_loss / total_samples), total_mae / total_samples, total_loss / total_samples

@torch.no_grad()
def test(loader):
    model.eval()
    preds, actuals = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        preds.append(out.view(-1).cpu().numpy())
        actuals.append(data.y.view(-1).cpu().numpy())
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    mse = mean_squared_error(actuals, preds)
    return np.sqrt(mse), mean_absolute_error(actuals, preds), mse

# -------------------- TRAINING LOOP ------------------------
os.makedirs('models', exist_ok=True)
best_rmse = float('inf')
epochs = 300
patience = 20
counter = 0

model.reset_parameters()
for epoch in range(epochs):
    tr_rmse, tr_mae, _ = train(train_loader)
    te_rmse, te_mae, _ = test(test_loader)
    print(f"Epoch {epoch+1}/{epochs}, Train RMSE: {tr_rmse:.4f}, Test RMSE: {te_rmse:.4f}")

    if te_rmse < best_rmse:
        best_rmse = te_rmse
        counter = 0
        torch.save(model.state_dict(), 'models/best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print("⏹ Early stopping triggered.")
            break

# -------------------- EVALUATION ---------------------------
model.load_state_dict(torch.load('models/best_model.pth', weights_only=True))

@torch.no_grad()
def evaluate(loader):
    model.eval()
    preds, actuals = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        preds.append(out.cpu().numpy().flatten())
        actuals.append(data.y.cpu().numpy().flatten())
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    df_eval = pd.DataFrame({'pred': preds, 'actual': actuals})
    return df_eval, r2_score(actuals, preds), mean_squared_error(actuals, preds), mean_absolute_error(actuals, preds)

train_results, train_r2, train_mse_val, train_mae_val = evaluate(train_loader)
test_results, test_r2, test_mse_val, test_mae_val = evaluate(test_loader)

# -------------------- FULL PREDICTION SAVE -----------------
@torch.no_grad()
def predict_all(loader):
    model.eval()
    preds = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        preds.append(out.cpu().numpy().flatten())
    return np.concatenate(preds)

full_loader = DataLoader(graph_list, batch_size=32, shuffle=False)
preds_all = predict_all(full_loader)

df_save = pd.DataFrame({
    "smiles": df_final['smiles'],
    "actual_exp": df_final['exp'],        # keep experimental value as-is
    "pred_exp": preds_all,
})
df_save["residual"] = df_save["pred_exp"] - df_save["actual_exp"]
df_save.to_csv("all_predicted_exp_values.csv", index=False)

print(f"\n✅ Saved all {len(df_save)} predictions to all_predicted_exp_values.csv")

# -------------------- PLOTTING -----------------------------
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelweight': 'bold',
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'axes.linewidth': 1.5,
    'lines.linewidth': 2
})

os.makedirs('plots', exist_ok=True)
plt.figure(figsize=(14, 6), dpi=300)

# Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(test_results['actual'], test_results['pred'], color='blue', label='Test', edgecolors='black', s=50)
sns.regplot(data=train_results, x='actual', y='pred', color='red',
            scatter_kws={'s': 40, 'alpha': 0.3, 'edgecolor': 'black'})
plt.xlabel('Actual', fontsize=20, fontweight='bold')
plt.ylabel('Predicted', fontsize=20, fontweight='bold')
plt.legend(['Test', 'Train'], frameon=False, loc='lower right', fontsize=20)
anchored_text = AnchoredText(f"R² (Train): {train_r2:.4f}\nR² (Test): {test_r2:.4f}",
                             loc='upper left', prop=dict(size=18, weight='bold'))
plt.gca().add_artist(anchored_text)
plt.grid(True, linestyle='--', alpha=0.5)

# Residual plot
plt.subplot(1, 2, 2)
plt.scatter(test_results['actual'], test_results['actual'] - test_results['pred'],
            color='blue', label='Test', edgecolors='black', s=50)
sns.scatterplot(x=train_results['actual'], y=train_results['actual'] - train_results['pred'],
                color='red', alpha=0.3, edgecolor='black', label='Train')
plt.xlabel('Actual', fontsize=20, fontweight='bold')
plt.ylabel('Residual (Actual - Predicted)', fontsize=20, fontweight='bold')
plt.legend(loc='upper left', frameon=False, fontsize=20)
plt.grid(True, linestyle='--', alpha=0.5)
ax = plt.gca()
ax.text(0.99, 0.03, 'Attentive FP', transform=ax.transAxes,
        fontsize=18, fontweight='bold', color='black',
        va='bottom', ha='right', alpha=0.9)
plt.tight_layout()
plt.savefig('plots/act_pred_res_plot_hd.png', dpi=600)
plt.close()

# -------------------- FINAL METRICS ------------------------
print("\n📘 Final Performance Metrics")
print("-" * 40)
print(f"Train MAE   : {train_mae_val:.4f}")
print(f"Test  MAE   : {test_mae_val:.4f}")
print(f"Train MSE   : {train_mse_val:.4f}")
print(f"Test  MSE   : {test_mse_val:.4f}")
print(f"Train RMSE  : {np.sqrt(train_mse_val):.4f}")
print(f"Test  RMSE  : {np.sqrt(test_mse_val):.4f}")
print(f"Train R²    : {train_r2:.4f}")
print(f"Test  R²    : {test_r2:.4f}")
print("-" * 40)
print(f"Execution time: {(time.time() - start_time)/60:.2f} minutes")
print("Plots saved in: plots/act_pred_res_plot_hd.png")
print("All predictions saved in: all_predicted_exp_values.csv")

