# ================================
# Scaffold Split Analysis Script
# ================================

import random
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt

# ----------------
# Reproducibility
# ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ----------------
# Load dataset
# ----------------
df = pd.read_csv('/home/agbande-remote/sandeep/gnn/afp/afp_final/eql_2_gcn/Lipophilicity.csv')
df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
df = df.dropna(subset=['mol'])
df['molwt'] = df['mol'].apply(Descriptors.MolWt)
df = df[df['molwt'] <= 700].reset_index(drop=True)

# ----------------
# Scaffold utilities
# ----------------
def generate_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

# ----------------
# Scaffold-based split
# ----------------
def scaffold_split(df, frac_train=0.8, seed=42):
    random.seed(seed)
    scaffold_dict = {}

    for idx, smi in enumerate(df['smiles']):
        scaf = generate_scaffold(smi)
        scaffold_dict.setdefault(scaf, []).append(idx)

    # Sort scaffolds by size (largest first)
    scaffolds_sorted = sorted(
        scaffold_dict.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    train_cutoff = int(len(df) * frac_train)
    train_idx, test_idx = [], []
    total = 0

    for scaffold, indices in scaffolds_sorted:
        if total < train_cutoff:
            train_idx.extend(indices)
        else:
            test_idx.extend(indices)
        total += len(indices)

    return train_idx, test_idx, scaffold_dict

train_idx, test_idx, scaffold_dict = scaffold_split(df)

# ----------------
# Basic counts
# ----------------
train_smiles = df.loc[train_idx, 'smiles']
test_smiles  = df.loc[test_idx, 'smiles']

train_scaffolds = set(train_smiles.apply(generate_scaffold))
test_scaffolds  = set(test_smiles.apply(generate_scaffold))

# ----------------
# Reviewer-critical checks
# ----------------
print("\n🔍 Scaffold Leakage Check")
print("-" * 40)
print("Overlapping scaffolds:", len(train_scaffolds & test_scaffolds))
assert len(train_scaffolds & test_scaffolds) == 0, "Scaffold leakage detected!"

# ----------------
# Scaffold statistics
# ----------------
train_scaffold_sizes = [
    sum(df['smiles'].apply(generate_scaffold) == scaf)
    for scaf in train_scaffolds
]

test_scaffold_sizes = [
    sum(df['smiles'].apply(generate_scaffold) == scaf)
    for scaf in test_scaffolds
]

# ----------------
# Table for manuscript / reviewers
# ----------------
summary_table = pd.DataFrame({
    'Split': ['Train', 'Test'],
    'Num Molecules': [len(train_idx), len(test_idx)],
    'Num Unique Scaffolds': [len(train_scaffolds), len(test_scaffolds)],
    'Avg Molecules per Scaffold': [
        np.mean(train_scaffold_sizes),
        np.mean(test_scaffold_sizes)
    ],
    'Median Molecules per Scaffold': [
        np.median(train_scaffold_sizes),
        np.median(test_scaffold_sizes)
    ],
    'Max Molecules per Scaffold': [
        np.max(train_scaffold_sizes),
        np.max(test_scaffold_sizes)
    ]
})

print("\n📊 Scaffold Split Summary Table")
print("-" * 40)
print(summary_table)

# Save table
summary_table.to_csv("scaffold_split_summary.csv", index=False)

# ----------------
# Target distribution comparison
# ----------------
plt.figure(figsize=(7, 5))
plt.hist(df.loc[train_idx, 'exp'], bins=30, alpha=0.7, label='Train')
plt.hist(df.loc[test_idx, 'exp'], bins=30, alpha=0.7, label='Test')
plt.xlabel('Target Value')
plt.ylabel('Frequency')
plt.title('Target Distribution: Scaffold Split')
plt.legend()
plt.tight_layout()
plt.savefig("scaffold_target_distribution.png", dpi=300)
plt.close()

print("\n✅ Analysis complete.")
print("Saved:")
print(" - scaffold_split_summary.csv")
print(" - scaffold_target_distribution.png")

