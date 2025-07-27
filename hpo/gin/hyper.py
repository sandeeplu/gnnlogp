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
from torch_geometric.nn import GINConv, global_mean_pool
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx, from_smiles

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

# Define the GIN model
class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GIN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )))
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )))
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

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

from optuna.trial import TrialState
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
    model = GIN(in_channels=9, hidden_channels=hidden_channels, out_channels=1,
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
    study_start_time = time.time()  # Start time for the entire optimization

    study = optuna.create_study(direction='minimize', study_name='hyperparameter-tune-gin', storage='sqlite:///htune_gin.db')
    study.optimize(objective, n_trials=500)

    # Print and display final summary
    summarize_study(study, study_start_time)

def summarize_study(study, study_start_time):
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Best metrics across all trials
    best_r2_trial = max(complete_trials, key=lambda t: t.user_attrs.get("r2", float("-inf")))
    best_mae_trial = min(complete_trials, key=lambda t: t.user_attrs.get("mae", float("inf")))
    best_mse_trial = min(complete_trials, key=lambda t: t.user_attrs.get("mse", float("inf")))
    best_rmse_trial = study.best_trial

    print("\n=== Study Summary ===")
    print(f"Total Trials: {len(study.trials)}")
    print(f"Pruned Trials: {len(pruned_trials)}")
    print(f"Completed Trials: {len(complete_trials)}")
    print(f"Optimization Duration: {time.time() - study_start_time:.2f} seconds\n")

    # Print the best trial for each metric
    print("Best Metrics from Completed Trials:")
    print(f"  Best R²: {best_r2_trial.user_attrs['r2']:.4f}, Hyperparameters: {best_r2_trial.params}")
    print(f"  Best MAE: {best_mae_trial.user_attrs['mae']:.4f}, Hyperparameters: {best_mae_trial.params}")
    print(f"  Best MSE: {best_mse_trial.user_attrs['mse']:.4f}, Hyperparameters: {best_mse_trial.params}")
    print(f"  Best RMSE (Objective): {best_rmse_trial.value:.4f}, Hyperparameters: {best_rmse_trial.params}")

    # Print all completed trials in a tabular format
    print("\n=== Completed Trials Details ===")
    print(f"{'Trial':<10}{'R²':<10}{'MAE':<10}{'MSE':<10}{'RMSE':<10}{'Params'}")
    print("=" * 80)
    for i, trial in enumerate(complete_trials):
        print(
            f"{i:<10}{trial.user_attrs.get('r2', 'N/A'):<10.4f}"
            f"{trial.user_attrs.get('mae', 'N/A'):<10.4f}"
            f"{trial.user_attrs.get('mse', 'N/A'):<10.4f}"
            f"{trial.value:<10.4f}"
            f"{trial.params}"
        )

# Main execution
if __name__ == "__main__":
    start_time = time.time()  # Start time for the script

    seed_set(42)
    run_optimization()

    end_time = time.time()  # End time for the script
    total_duration = end_time - start_time
    print(f"\nTotal execution time: {total_duration:.2f} seconds")

