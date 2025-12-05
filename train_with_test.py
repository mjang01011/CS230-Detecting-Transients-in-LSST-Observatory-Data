import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from lib.sequence_dataset import SequenceDataset
from models.tcn import LightCurveTCN
from sklearn.metrics import f1_score


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# input_csv = 'data/output/processed_training_asinh_norm.csv'
input_csv = 'data/output/processed_training_gp_asinh.csv'
max_length = 200
batch_size = 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'
df = pd.read_csv(input_csv)
print(f"Loaded {len(df)} observations for {df['object_id'].nunique()} objects")
print(f"Columns: {list(df.columns)}")

object_ids = sorted(df['object_id'].unique())
n_objects = len(object_ids)
passbands = sorted(df['passband'].unique())
n_passbands = len(passbands)

print(f"Converting {n_objects} objects to sequence format...")
print(f"Passbands: {passbands}")

data = np.zeros((n_objects, max_length, n_passbands), dtype=np.float32)
labels = np.zeros(n_objects, dtype=np.int64)

for idx, obj_id in enumerate(tqdm(object_ids, desc="Processing objects")):
    obj_data = df[df['object_id'] == obj_id]
    labels[idx] = obj_data['target'].iloc[0]

    for pb_idx, pb in enumerate(passbands):
        pb_data = obj_data[obj_data['passband'] == pb].sort_values('t_centered')
        if len(pb_data) > 0:
            times = pb_data['t_centered'].values
            flux = pb_data['flux'].values
            for t, f in zip(times, flux):
                idx_t = int(t + max_length // 2)
                if 0 <= idx_t < max_length:
                    data[idx, idx_t, pb_idx] = f

print(f"\nData shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

unique_labels = np.unique(labels)
label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}
labels_remapped = np.array([label_mapping[label] for label in labels])
num_classes = len(unique_labels)

print(f"Label mapping: {label_mapping}")
print(f"Number of classes: {num_classes}")

X_temp, X_test, y_temp, y_test = train_test_split(
    data, labels_remapped,
    test_size=0.1, random_state=42, stratify=labels_remapped
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.1111, random_state=42, stratify=y_temp
)

print(f"Train: {len(y_train)} ({len(y_train)/len(labels_remapped)*100:.1f}%)")
print(f"Val: {len(y_val)} ({len(y_val)/len(labels_remapped)*100:.1f}%)")
print(f"Test: {len(y_test)} ({len(y_test)/len(labels_remapped)*100:.1f}%)")

train_dataset = SequenceDataset(X_train, y_train)
val_dataset = SequenceDataset(X_val, y_val)
test_dataset = SequenceDataset(X_test, y_test)

def show_split_statistics(y_train, y_val, y_test):
    """
    Display class frequencies and percentages across train/val/test splits.
    """
    classes = np.unique(np.concatenate([y_train, y_val, y_test]))

    results = []
    for cls in classes:
        train_count = np.sum(y_train == cls)
        val_count = np.sum(y_val == cls)
        test_count = np.sum(y_test == cls)
        total = train_count + val_count + test_count

        results.append({
            'Class': int(cls),
            'Train': train_count,
            'Train %': f"{100 * train_count / len(y_train):.2f}%",
            'Val': val_count,
            'Val %': f"{100 * val_count / len(y_val):.2f}%",
            'Test': test_count,
            'Test %': f"{100 * test_count / len(y_test):.2f}%",
            'Total': total
        })

    df = pd.DataFrame(results)

    totals = {
        'Class': 'TOTAL',
        'Train': len(y_train),
        'Train %': '100.00%',
        'Val': len(y_val),
        'Val %': '100.00%',
        'Test': len(y_test),
        'Test %': '100.00%',
        'Total': len(y_train) + len(y_val) + len(y_test)
    }
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    print("\n" + "="*80)
    print("TRAIN/VAL/TEST SPLIT STATISTICS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    print(f"Split ratio: {len(y_train)}/{len(y_val)}/{len(y_test)} = "
          f"{100*len(y_train)/(len(y_train)+len(y_val)+len(y_test)):.1f}%/"
          f"{100*len(y_val)/(len(y_train)+len(y_val)+len(y_test)):.1f}%/"
          f"{100*len(y_test)/(len(y_train)+len(y_val)+len(y_test)):.1f}%")
    print("="*80 + "\n")

    return df

split_df = show_split_statistics(y_train, y_val, y_test)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for sequences, labels in dataloader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds), all_preds, all_labels

def train_with_params(lr, 
                      #weight_decay,
                      num_epochs=50):
    model = LightCurveTCN(
        input_size=6,
        num_classes=14,
        hidden_size=64,
        num_layers=7,
        kernel_size=3,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

    return model, best_val_loss, train_losses, val_losses

results = []
config_idx = 0

model, val_loss, train_losses, val_losses = train_with_params(0.001, num_epochs=30)


print("\n" + "="*80)
print("Best Configuration:")
print(f"Best Val Loss: {val_loss:.4f}")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
criterion = nn.CrossEntropyLoss()

test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
macro_f1 = f1_score(test_labels, test_preds, average='macro')
print("\nTest Set Performance:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print("\nClassification Report:")
print(classification_report(test_labels, test_preds,
                          target_names=[f"Class {reverse_label_mapping[i]}" for i in range(num_classes)]))

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(train_losses, label='Train Loss')
ax.plot(val_losses, label='Val Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title(f'')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('training_curves_tcn_raw_best.png', dpi=300)

cm_test = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f"Class {reverse_label_mapping[i]}" for i in range(num_classes)],
            yticklabels=[f"Class {reverse_label_mapping[i]}" for i in range(num_classes)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix - TCN')
plt.tight_layout()
plt.savefig('confusion_matrix_rnn_raw_test.png', dpi=300)