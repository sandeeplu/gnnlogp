import os
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_smiles
from torch_geometric.nn import GIN
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import r2_score
from torch_geometric.nn import GINConv
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

# Load and preprocess dataset
df = pd.read_csv('/home/agbande-remote/sandeep/gnn/afp/afp_final/eql_2_gcn/Lipophilicity.csv')
df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
df = df.dropna(subset=['mol'])
df['molwt'] = df['mol'].apply(Descriptors.MolWt)
df_final = df[df['molwt'] <= 700].reset_index(drop=True)
df_final['labels'] = df_final['exp'].apply(lambda x: int(x > 2.36))

print(f"Binary class counts: {Counter(df_final['labels'])}")

# Graph dataset
graph_list = []
for i, smile in enumerate(df_final['smiles']):
    g = from_smiles(smile)
    g.x = g.x.float()
    g.y = torch.tensor([df_final['labels'][i]], dtype=torch.float)
    graph_list.append(g)

# Custom Dataset
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

# Train/test split
random.shuffle(graph_list)
train_size = int(0.8 * len(graph_list))
test_size = len(graph_list) - train_size
train_dataset, test_dataset = random_split(graph_list, [train_size, test_size], generator=generator)

# Dataloaders
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
torch.cuda.empty_cache()

# Training
def train(loader):
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
    acc = total_correct / total_samples
    return total_loss / total_samples, acc

# Testing
@torch.no_grad()
def test(loader):
    model.eval()
    total_loss = total_correct = total_samples = 0
    for data in loader:
        data = data.to(device)
        out = torch.sigmoid(model(data.x, data.edge_index, data.batch))
        loss = F.binary_cross_entropy(out, data.y.view(-1, 1))
        total_loss += loss.item() * data.num_graphs
        preds = (out > 0.5).float()
        total_correct += (preds == data.y.view(-1, 1)).sum().item()
        total_samples += data.num_graphs
    acc = total_correct / total_samples
    return total_loss / total_samples, acc

# Training loop
os.makedirs('models', exist_ok=True)
best_loss = float('inf')
epochs = 300
patience = 20
counter = 0
model.reset_parameters()

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in range(epochs):
    tr_loss, tr_acc = train(train_loader)
    te_loss, te_acc = test(test_loader)
    train_losses.append(tr_loss)
    train_accuracies.append(tr_acc)
    test_losses.append(te_loss)
    test_accuracies.append(te_acc)
    print(f"Epoch {epoch+1}/{epochs}, Train Acc: {tr_acc:.4f}, Test Acc: {te_acc:.4f}")

    if te_loss < best_loss:
        best_loss = te_loss
        counter = 0
        torch.save(model.state_dict(), 'models/best_model_cls.pth')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# Final Evaluation
#model.load_state_dict(torch.load('models/best_model_cls.pth'))
model.load_state_dict(torch.load('models/best_model_cls.pth', weights_only=True))
@torch.no_grad()
def evaluate(loader):
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
    return acc, prec, rec, f1, all_preds, all_labels

train_acc, train_prec, train_rec, train_f1, _, _ = evaluate(train_loader)
test_acc, test_prec, test_rec, test_f1, test_preds, test_labels = evaluate(test_loader)

# Plotting
os.makedirs('plots', exist_ok=True)
mpl.rcParams.update({
    'font.family': 'serif', 'font.size': 14, 'axes.labelweight': 'bold',
    'axes.titlesize': 16, 'axes.labelsize': 16, 'xtick.labelsize': 14,
    'ytick.labelsize': 14, 'legend.fontsize': 14, 'figure.dpi': 300,
    'savefig.dpi': 600, 'axes.linewidth': 1.5, 'lines.linewidth': 2
})

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Train Accuracy', color='red')
plt.plot(range(1, len(test_accuracies)+1), test_accuracies, label='Test Accuracy', color='blue')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend(frameon=False); plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout(); plt.savefig('plots/epoch_vs_accuracy.png'); plt.close()

# Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(test_labels, test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png')
plt.close()

"""
# Final Metrics
print("\n📘 Final Classification Metrics")
print("-" * 40)
print(f"Train Accuracy : {train_acc:.4f}")
print(f"Train Precision: {train_prec:.4f}")
print(f"Train Recall   : {train_rec:.4f}")
print(f"Train F1 Score : {train_f1:.4f}")
print(f"Test Accuracy  : {test_acc:.4f}")
print(f"Test Precision : {test_prec:.4f}")
print(f"Test Recall    : {test_rec:.4f}")
print(f"Test F1 Score  : {test_f1:.4f}")
print("-" * 40)
print(f"Execution time: {(time.time() - start_time)/60:.2f} minutes")
"""
# Final summary
print("\n Final Classification Metrics")
print("-" * 40)
print(f"Train Accuracy : {train_acc:.4f}")
print(f"Train Precision: {train_prec:.4f}")
print(f"Train Recall   : {train_rec:.4f}")
print(f"Train F1 Score : {train_f1:.4f}")
print(f"Train Loss     : {train_losses[-1]:.4f}")
print("-" * 40)
print(f"Test Accuracy  : {test_acc:.4f}")
print(f"Test Precision : {test_prec:.4f}")
print(f"Test Recall    : {test_rec:.4f}")
print(f"Test F1 Score  : {test_f1:.4f}")
print(f"Test Loss      : {test_losses[-1]:.4f}")
print("-" * 40)
print(f"Execution time : {(time.time() - start_time)/60:.2f} minutes")
print("📁 Metrics and plots saved in 'plots/' folder")



