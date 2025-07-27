import torch
import optuna
import random
import numpy as np
import pandas as pd
import os
import time
from math import sqrt
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_smiles
from torch.utils.data import random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rdkit import Chem
from rdkit.Chem import Descriptors
from optuna.trial import TrialState

# Reproducibility
def seed_set(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_set(42)

def prepare_data():
    df = pd.read_csv('/home/agbande-remote/sandeep/gnn/afp/afp_final/eql_2_gcn/Lipophilicity.csv')
    df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    df = df.dropna(subset=['mol'])
    df['molwt'] = df['mol'].apply(Descriptors.MolWt)
    df_final = df[df['molwt'] <= 700].reset_index(drop=True)
    graph_list = []
    for i, smile in enumerate(df_final['smiles']):
        g = from_smiles(smile)
        g.x = g.x.float()
        g.y = torch.tensor([df_final['exp'][i]], dtype=torch.float)
        graph_list.append(g)
    return graph_list

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.fc = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.fc.reset_parameters()
    
    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

# Training Loop
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss, total_examples = 0, 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        # Ensuring both inputs to mse_loss are [batch_size, 1]
        loss = torch.nn.functional.mse_loss(out, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        total_examples += data.num_graphs
    return sqrt(total_loss / total_examples)

@torch.no_grad()
def test(model, test_loader, device):
    model.eval()
    true_values, predictions = [], []
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        predictions.append(out.view(-1).cpu().numpy())
        true_values.append(data.y.view(-1).cpu().numpy())
    true_values = np.concatenate(true_values, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = sqrt(mse)
    r2 = r2_score(true_values, predictions)
    return mae, mse, rmse, r2

# Optuna Objective
def objective(trial):
    start_time = time.time()
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.6, step=0.1)
    num_layers = trial.suggest_int('num_layers', 2, 6)
    hidden_channels = trial.suggest_int('hidden_channels', 32, 192, step=32)
    batch_size = trial.suggest_int('batch_size', 16, 128, step=16)

    graph_list = prepare_data()
    dataset_size = len(graph_list)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    generator1 = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(graph_list, [train_size, test_size], generator=generator1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(
        in_channels=9,  # Change if your x features have different dimensionality
        hidden_channels=hidden_channels,
        out_channels=1,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(100):
        train(model, train_loader, optimizer, device)
        mae, mse, test_rmse, test_r2 = test(model, test_loader, device)
        trial.report(test_rmse, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    trial.set_user_attr("mae", float(mae))
    trial.set_user_attr("mse", float(mse))
    trial.set_user_attr("r2", float(test_r2))
    print(f"Trial completed in {time.time() - start_time:.2f} s | MAE: {mae:.4f} | RMSE: {test_rmse:.4f} | R²: {test_r2:.4f}")
    return test_rmse

# Study summary printout
def summarize_study(study, study_start_time):
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    best_r2_trial = max(complete_trials, key=lambda t: t.user_attrs.get("r2", float("-inf")))
    best_mae_trial = min(complete_trials, key=lambda t: t.user_attrs.get("mae", float("inf")))
    best_mse_trial = min(complete_trials, key=lambda t: t.user_attrs.get("mse", float("inf")))
    best_rmse_trial = study.best_trial

    print("\n=== Study Summary ===")
    print(f"Total Trials: {len(study.trials)}")
    print(f"Pruned Trials: {len(pruned_trials)}")
    print(f"Completed Trials: {len(complete_trials)}")
    print(f"Optimization Duration: {time.time() - study_start_time:.2f} seconds\n")

    print("Best Metrics from Completed Trials:")
    print(f"  Best R² : {best_r2_trial.user_attrs['r2']:.4f}, Hyperparameters: {best_r2_trial.params}")
    print(f"  Best MAE: {best_mae_trial.user_attrs['mae']:.4f}, Hyperparameters: {best_mae_trial.params}")
    print(f"  Best MSE: {best_mse_trial.user_attrs['mse']:.4f}, Hyperparameters: {best_mse_trial.params}")
    print(f"  Best RMSE (Objective): {best_rmse_trial.value:.4f}, Hyperparameters: {best_rmse_trial.params}")

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

def run_optimization():
    study_start_time = time.time()
    study = optuna.create_study(
        direction='minimize', 
        study_name='hyperparameter-tune-gcn',
        storage='sqlite:///htune_gcn.db'
    )
    study.optimize(objective, n_trials=500)  # Increase n_trials as desired
    summarize_study(study, study_start_time)

if __name__ == "__main__":
    seed_set(42)
    run_optimization()

