import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_smiles
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import SubsetRandomSampler
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx, from_smiles

# Reproducibility
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

# Load dataset
#df_final = pd.read_csv('/home/agbande-remote/sandeep/gnn/afp/afp_final/eql_2_gcn/Lipophilicity.csv')
df_final = pd.read_csv('/home/agbande-remote/sandeep/gnn/fc/Lipophilicity_final.csv')
# Prepare graph dataset
graph_list = []
for i, smile in enumerate(df_final['smiles']):
    g = from_smiles(smile)
    g.x = g.x.float()
    g.y = torch.tensor([df_final['exp'][i]], dtype=torch.float)
    g.smiles = smile
    graph_list.append(g)

# Define GCN model with num_layers and dropout
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=6, dropout=0.1):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Create layers dynamically based on num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)  # Global pooling
        x = self.lin(x)
        return x

    # Reset parameters method
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()


# Training and evaluation
def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.mse_loss(out, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return np.sqrt(total_loss / len(loader.dataset))

@torch.no_grad()
def test(model, loader):
    model.eval()
    preds, targets = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        preds.append(out.cpu())
        targets.append(data.y.view(-1, 1).cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, preds)
    return rmse, mae, mse

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        preds.append(out.cpu())
        targets.append(data.y.view(-1, 1).cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    r2 = r2_score(targets, preds)
    return r2

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# K-Fold Cross Validation
kfold = KFold(n_splits=10, shuffle=True, random_state=50)
results_r2, results_rmse, results_mae, results_mse = [], [], [], []
overall_start_time = time.time()

for fold, (train_ids, test_ids) in enumerate(kfold.split(graph_list)):
    print(f"\nFOLD {fold} ----------------------------")
    fold_start_time = time.time()

    train_loader = DataLoader(graph_list, batch_size=16, sampler=SubsetRandomSampler(train_ids))
    test_loader = DataLoader(graph_list, batch_size=16, sampler=SubsetRandomSampler(test_ids))

    # Define model and optimizer
    model = GCN(in_channels=9, hidden_channels=192, out_channels=1,
                     num_layers=5, dropout=0.2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr= 0.0004407086844982349,
                             weight_decay=0.000004856076517711844)
    model.reset_parameters()
#early stop
    best_rmse = float('inf')
    patience, trigger_times = 30, 0
    best_model_state = None

    for epoch in range(300):
        train_rmse = train(model, train_loader, optimizer)
        val_rmse, val_mae, val_mse = test(model, test_loader)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            trigger_times = 0
            best_model_state = model.state_dict()
        else:
            trigger_times += 1

        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}: Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}')

        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

    # Load best model and evaluate
    model.load_state_dict(best_model_state)
    rmse, mae, mse = test(model, test_loader)
    r2 = evaluate(model, test_loader)

    results_r2.append(r2)
    results_rmse.append(rmse)
    results_mae.append(mae)
    results_mse.append(mse)

    print(f"Fold {fold} R2 score: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")
    print(f"Fold {fold} completed in {(time.time() - fold_start_time):.2f} seconds.")

# Final summary
print("\nFinal Cross-Validation Results:")
print(f"Average R2:  {np.mean(results_r2):.4f} ± {np.std(results_r2):.4f}")
print(f"Average RMSE: {np.mean(results_rmse):.4f} ± {np.std(results_rmse):.4f}")
print(f"Average MAE:  {np.mean(results_mae):.4f} ± {np.std(results_mae):.4f}")
print(f"Average MSE:  {np.mean(results_mse):.4f} ± {np.std(results_mse):.4f}")
print(f"Total execution time: {(time.time() - overall_start_time):.2f} seconds")

