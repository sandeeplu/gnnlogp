import os
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from math import sqrt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.utils.data import random_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GIN
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import r2_score
from torch_geometric.nn import GINConv
from torch_geometric.utils import to_networkx, from_smiles
from matplotlib.offsetbox import AnchoredText
from scipy.signal import savgol_filter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GINConv, MLP, global_add_pool
# Set up
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

# Load and filter dataset
df_final = pd.read_csv('/home/agbande-remote/sandeep/gnn/afp/afp_final/eql_2_gcn/Lipophilicity.csv')
#df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
#df = df.dropna(subset=['mol'])
#df['molwt'] = df['mol'].apply(Descriptors.MolWt)
#df_final = df[df['molwt'] <= 700].reset_index(drop=True)

# Visualize a sample graph
smile = df_final['smiles'][92]
g = from_smiles(smile, with_hydrogen=False)
G = to_networkx(g)
nx.draw(G, with_labels=True, font_weight='bold')
plt.savefig("G.png")

# Graph dataset preparation
graph_list = []
for i, smile in enumerate(df_final['smiles']):
    g = from_smiles(smile)
    g.x = g.x.float()
    g.y = torch.tensor([df_final['exp'][i]], dtype=torch.float)
    graph_list.append(g)

# Custom Dataset Class
class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self): return 'Lipophilicity.csv'
    @property
    def processed_file_names(self): return 'data.dt'
    def download(self): pass

    def process(self):
        data_list = graph_list
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.save(data_list, self.processed_paths[0])

lipo = MyOwnDataset(root='.')

# Dataset splitting
random.shuffle(graph_list)
train_size = int(0.8 * len(graph_list))
test_size = len(graph_list) - train_size
train_dataset, test_dataset = random_split(graph_list, [train_size, test_size], generator=generator)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=generator)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define GIN Model
# GIN Model
class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.4)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.mlp(x)

# Define model and optimizer
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN(in_channels=9, hidden_channels=256, out_channels=1, num_layers=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=4.4755e-6, weight_decay=9.8267e-6)

##Use a Learning Rate Scheduler

#from torch.optim.lr_scheduler import ReduceLROnPlateau
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
# Training function
def train(loader):
    model.train()
    total_loss = total_mae = total_mse = total_samples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        mse_loss = F.mse_loss(out, data.y.view(-1, 1))
        mae_loss = F.l1_loss(out, data.y.view(-1, 1))
        mse_loss.backward()
        optimizer.step()
        total_loss += mse_loss.item() * data.num_graphs
        total_mae += mae_loss.item() * data.num_graphs
        total_samples += data.num_graphs
    return np.sqrt(total_loss / total_samples), total_mae / total_samples, total_loss / total_samples

# Testing function
@torch.no_grad()
def test(loader):
    model.eval()
    preds, actuals = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        preds.append(out.view(-1).cpu().numpy())
        actuals.append(data.y.view(-1).cpu().numpy())
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    mse = mean_squared_error(actuals, preds)
    return np.sqrt(mse), mean_absolute_error(actuals, preds), mse

#Check exit dir

os.makedirs('models', exist_ok=True)

# Training loop with Early Stopping
best_rmse = float('inf')
epochs = 300
patience = 20
counter = 0
model.reset_parameters()
train_rmse, train_mae, train_mse = [], [], []
test_rmse, test_mae, test_mse = [], [], []

for epoch in range(epochs):
    tr_rmse, tr_mae, tr_mse = train(train_loader)
    te_rmse, te_mae, te_mse = test(test_loader)
    train_rmse.append(tr_rmse)
    train_mae.append(tr_mae)
    train_mse.append(tr_mse)
    test_rmse.append(te_rmse)
    test_mae.append(te_mae)
    test_mse.append(te_mse)
    print(f"Epoch {epoch+1}/{epochs}, Train RMSE: {tr_rmse:.4f}, Test RMSE: {te_rmse:.4f}")

    if te_rmse < best_rmse:
        best_rmse = te_rmse
        counter = 0
        torch.save(model.state_dict(), 'models/best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# Load best model for evaluation
model.load_state_dict(torch.load('models/best_model.pth'))

@torch.no_grad()
def evaluate(loader):
    model.eval()
    results = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        results.append(torch.cat((out, data.y.view(-1, 1)), dim=1))
    results = torch.cat(results).cpu().numpy()
    df_eval = pd.DataFrame(results, columns=['pred', 'actual'])
    return df_eval, r2_score(df_eval['actual'], df_eval['pred']), mean_squared_error(df_eval['actual'], df_eval['pred']), mean_absolute_error(df_eval['actual'], df_eval['pred'])

train_results, train_r2, train_mse_val, train_mae_val = evaluate(train_loader)
test_results, test_r2, test_mse_val, test_mae_val = evaluate(test_loader)

# Plotting
mpl.rcParams.update({
    'font.family': 'serif', 'font.size': 14, 'axes.labelweight': 'bold',
    'axes.titlesize': 16, 'axes.labelsize': 16, 'xtick.labelsize': 14,
    'ytick.labelsize': 14, 'legend.fontsize': 14, 'figure.dpi': 300,
    'savefig.dpi': 600, 'axes.linewidth': 1.5, 'lines.linewidth': 2
})
os.makedirs('plots', exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_rmse)+1), train_rmse, label='Train RMSE', color='red')
plt.plot(range(1, len(test_rmse)+1), test_rmse, label='Test RMSE', color='blue')
plt.xlabel('Epoch'); plt.ylabel('RMSE')
plt.legend(frameon=False); plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout(); plt.savefig('plots/epoch_vs_rmse.png'); plt.close()

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(test_results['actual'], test_results['pred'], color='blue', label='Test', edgecolors='black', s=50)
sns.regplot(data=train_results, x='actual', y='pred', color='red', scatter_kws={'s':40, 'alpha':0.3, 'edgecolor':'black'})
plt.xlabel('Actual', weight='bold')
plt.ylabel('Predicted', weight='bold')
plt.legend(['Test', 'Train'], frameon=False, loc='lower right')
anchored_text = AnchoredText(f"R² (Train): {train_r2:.4f}\nR² (Test): {test_r2:.4f}", loc='upper left', prop=dict(size=12, weight='bold'))
plt.gca().add_artist(anchored_text)
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(1, 2, 2)
plt.scatter(test_results['actual'], test_results['actual'] - test_results['pred'], color='blue', label='Test', edgecolors='black', s=50)
sns.scatterplot(x=train_results['actual'], y=train_results['actual'] - train_results['pred'], color='red', alpha=0.3, edgecolor='black', label='Train')
plt.xlabel('Actual', weight='bold')
plt.ylabel('Residual (Actual - Predicted)', weight='bold')
plt.legend(loc='upper left', frameon=False)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/act_pred_res_plot.png')
plt.close()

plt.figure(figsize=(7, 6))
sns.histplot(test_results['actual'] - test_results['pred'], kde=True, color='blue', bins=30, edgecolor='black')
plt.xlabel('Prediction Error'); plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout(); plt.savefig('plots/error_histogram_kde_plot.png'); plt.close()

# Final Metrics
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

