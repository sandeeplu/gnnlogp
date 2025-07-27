import torch
import optuna
from optuna.trial import TrialState
import pickle
import numpy as np
import pandas as pd
import random
import os
from math import sqrt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import from_smiles
from torch_geometric.loader import DataLoader
import time  # Import time module to track execution time
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GraphSAGE
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx, from_smiles
from torch_geometric.nn import SAGEConv

# Check if distributed training is required (use multi-GPU if available)
def setup(rank, world_size):
    if world_size > 1:
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

def cleanup():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

# Set seed for reproducibility
def seed_set(seed=50):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_set(42)


# Load the dataset and prepare graphs
def prepare_data():
    df_final = pd.read_csv('/home/agbande-remote/sandeep/gnn/fc/Lipophilicity_final.csv')

    graph_list = []
    for i, smile in enumerate(df_final['smiles']):
        g = from_smiles(smile)
        g.x = g.x.float()
        y = torch.tensor(df_final['exp'][i], dtype=torch.float).view(1, -1)
        g.y = y
        graph_list.append(g)

    return graph_list

# Define GraphSAGE Model
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return global_mean_pool(x, batch)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

# Training loop
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    total_examples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        total_examples += data.num_graphs
    return sqrt(total_loss / total_examples)

# Test loop
@torch.no_grad()
def test(model, test_loader, device):
    model.eval()
    true_values = []
    predictions = []
    mse = []
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        true_values.append(data.y.cpu().numpy())
        predictions.append(out.cpu().numpy())
        l = F.mse_loss(out, data.y, reduction='none').cpu()
        mse.append(l)

    true_values = np.concatenate(true_values, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    mae = mean_absolute_error(true_values, predictions)
    mse_value = mean_squared_error(true_values, predictions)
    rmse = sqrt(mse_value)
    r2 = r2_score(true_values, predictions)

    return mae, mse_value, rmse, r2
# Objective function for Optuna optimization
def objective(trial):
    start_time = time.time()  # Start time for the trial

    # Hyperparameters to be tuned
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.6, step=0.1)
    num_layers = trial.suggest_int('num_layers', 2, 6)
    hidden_channels = trial.suggest_int('hidden_channels', 32, 192, step=32)
    batch_size = trial.suggest_int('batch_size', 16, 128, step=16)

    # Prepare dataset
    graph_list = prepare_data()
    train_ratio = 0.80
    dataset_size = len(graph_list)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    generator1 = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(graph_list, [train_size, test_size], generator=generator1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Check if CUDA is available and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the GraphSAGE model
    model = GraphSAGEModel(in_channels=9, hidden_channels=hidden_channels, out_channels=1,
                           num_layers=num_layers, dropout=dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(100):
        train_rmse = train(model, train_loader, optimizer, device)
        mae, mse_value, test_rmse, test_r2 = test(model, test_loader, device)

        trial.report(test_rmse, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()
    # Convert metrics to Python float for JSON serialization
    trial.set_user_attr("mae", float(mae))
    trial.set_user_attr("mse", float(mse_value))
    trial.set_user_attr("r2", float(test_r2))

    print(f"Trial completed in {time.time() - start_time:.2f} seconds")
    print(f"Trial MAE: {mae:.4f}, MSE: {mse_value:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
    return test_rmse


# Run Optuna study for hyperparameter optimization
def run_optimization():
    study_start_time = time.time()  # Start time for the optimization

    study = optuna.create_study(direction='minimize', study_name='hyperparameter-tune-gsage', storage='sqlite:///htune_gsage.db')
    study.optimize(objective, n_trials=200)

    print("\n=== Best Trial Summary ===")
    trial = study.best_trial
    print(f"Best RMSE (Value): {trial.value:.3f}")
    print(f"Best r2 (Value): {trial.value:.3f}")
    print("Best Hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    print("\n=== Study Statistics ===")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len(study.get_trials(states=[TrialState.PRUNED]))}")
    print(f"Number of completed trials: {len(study.get_trials(states=[TrialState.COMPLETE]))}")

    study_end_time = time.time()  # End time for the optimization
    print(f"Optimization completed in {study_end_time - study_start_time:.2f} seconds")

    return study


# Analyze completed trials
def analyze_trials(study):
    print("\n=== Completed Trials Summary ===")
    print(f"{'Trial':<6}{'R²':<8}{'MAE':<8}{'MSE':<8}{'RMSE':<8}{'Hyperparameters':<40}")
    print("=" * 80)
    complete_trials = study.get_trials(states=[TrialState.COMPLETE])
    for i, trial in enumerate(complete_trials):
        r2 = trial.user_attrs.get("r2", "N/A")
        mae = trial.user_attrs.get("mae", "N/A")
        mse = trial.user_attrs.get("mse", "N/A")
        rmse = trial.value
        params = trial.params
        print(f"{i:<6}{r2:<8.4f}{mae:<8.4f}{mse:<8.4f}{rmse:<8.4f}{params}")


# Main execution
if __name__ == "__main__":
    start_time = time.time()  # Start time for the script

    seed_set(42)
    study = run_optimization()
    analyze_trials(study)

    print(f"\nTotal script execution time: {time.time() - start_time:.2f} seconds")

