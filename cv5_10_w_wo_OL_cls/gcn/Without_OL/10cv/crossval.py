import os
import time
import random
import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_smiles
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import r2_score
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_networkx, from_smiles

# Set environment
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Seed for reproducibility
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

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess data
df = pd.read_csv('/home/agbande-remote/sandeep/gnn/fc/Lipophilicity_final.csv')
df['labels'] = df['exp'].apply(lambda x: int(x > 2.36))  # Binary classification
print(f"Class distribution: {Counter(df['labels'])}")

# Graph dataset
graph_list = []
for i, smile in enumerate(df['smiles']):
    try:
        g = from_smiles(smile)
        g.x = g.x.float()
        g.y = torch.tensor([df['labels'][i]], dtype=torch.float)
        g.smiles = smile
        graph_list.append(g)
    except Exception as e:
        print(f"Failed to convert {smile}: {e}")

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



# Training function
def train(model, loader, optimizer):
    model.train()
    total_loss = total_correct = total_samples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = torch.sigmoid(model(data.x, data.edge_index, data.batch))
        loss = F.binary_cross_entropy(out, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        preds = (out > 0.5).float()
        total_correct += (preds == data.y.view(-1, 1)).sum().item()
        total_samples += data.num_graphs
    return total_loss / total_samples, total_correct / total_samples

# Validation loss function (no optimizer, no gradients)
@torch.no_grad()
def compute_loss(model, loader):
    model.eval()
    total_loss = total_samples = 0
    for data in loader:
        data = data.to(device)
        out = torch.sigmoid(model(data.x, data.edge_index, data.batch))
        loss = F.binary_cross_entropy(out, data.y.view(-1, 1))
        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
    return total_loss / total_samples

# Evaluation function (metrics)
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for data in loader:
        data = data.to(device)
        out = torch.sigmoid(model(data.x, data.edge_index, data.batch))
        preds = (out > 0.5).float().cpu().numpy().flatten()
        labels = data.y.cpu().numpy().flatten()
        all_preds.extend(preds)
        all_labels.extend(labels)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, prec, rec, f1

# 5-Fold Stratified CV setup
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=50)
all_acc, all_prec, all_rec, all_f1 = [], [], [], []
overall_start = time.time()

labels_np = np.array([g.y.item() for g in graph_list])

for fold, (train_ids, test_ids) in enumerate(kfold.split(graph_list, labels_np)):
    print(f"\n🌀 Fold {fold} -----------------------------")
    fold_start = time.time()

    train_loader = DataLoader([graph_list[i] for i in train_ids], batch_size=32, shuffle=True, generator=generator)
    test_loader = DataLoader([graph_list[i] for i in test_ids], batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Define model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(in_channels=9, hidden_channels=192, out_channels=1,
                     num_layers=5, dropout=0.2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr= 0.0004407086844982349,
                             weight_decay=0.000004856076517711844)

    model.reset_parameters()
   ## early stop

    best_loss = float('inf')
    patience, trigger_times = 30, 0
    best_model_state = None

    for epoch in range(300):
        train_loss, train_acc = train(model, train_loader, optimizer)
        val_loss = compute_loss(model, test_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
            trigger_times = 0
        else:
            trigger_times += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: Train Acc = {train_acc:.4f}, Val Loss = {val_loss:.4f}")

        if trigger_times >= patience:
            print("⏹️ Early stopping triggered.")
            break

    # Final evaluation
    model.load_state_dict(best_model_state)
    acc, prec, rec, f1 = evaluate(model, test_loader)
    all_acc.append(acc)
    all_prec.append(prec)
    all_rec.append(rec)
    all_f1.append(f1)

    print(f"✅ Fold {fold} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"Fold time: {time.time() - fold_start:.2f}s")

# Final summary
print("\n🎯 Final Cross-Validation Results:")
print(f"Avg Accuracy : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
print(f"Avg Precision: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}")
print(f"Avg Recall   : {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")
print(f"Avg F1 Score : {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
print(f"Total time: {(time.time() - overall_start):.2f}s")

