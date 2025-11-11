import os
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
from sklearn.metrics import f1_score
import random

from KAN_SleepNet import KanSleepNet

plt.rcParams.update({'font.size': 13})

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

gpu_index = 1
device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# input data
data_path = 'data/ISRUC_S1'

# output file name
file_name = 'KAN_SleepNet_ISRUC_S1_'

fnames = sorted(glob(os.path.join(data_path, '*.npz')))

subjects_X, subjects_y = [], []
for fname in fnames:
    samples = np.load(fname)
    x = samples['x']
    y = samples['y']

    # one-hot encoding
    y_oh = np.zeros((len(y), 5))
    for i, label in enumerate(y):
        y_oh[i, label] = 1.

    seq_length = 15
    X_seq, y_seq = [], []
    for j in range(0, len(x), seq_length):
        if j + seq_length < len(x):
            X_seq.append(x[j:j + seq_length])
            y_seq.append(y_oh[j:j + seq_length])
    subjects_X.append(np.array(X_seq))
    subjects_y.append(np.array(y_seq))

n_subjects = len(subjects_X)
all_indices = np.arange(n_subjects)

train_idx, test_idx = train_test_split(all_indices, test_size=0.15, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=15/85, random_state=42)

def concat_subjects(indices):
    X_cat = np.concatenate([subjects_X[i] for i in indices], axis=0)
    y_cat = np.concatenate([subjects_y[i] for i in indices], axis=0)
    return X_cat, y_cat

X_seq_train, y_seq_train = concat_subjects(train_idx)
X_seq_val, y_seq_val = concat_subjects(val_idx)
X_seq_test, y_seq_test = concat_subjects(test_idx)

X_seq_train = np.expand_dims(X_seq_train, 2)
X_seq_val = np.expand_dims(X_seq_val, 2)
X_seq_test = np.expand_dims(X_seq_test, 2)

X_seq_train = torch.tensor(X_seq_train, dtype=torch.float32).to(device)
X_seq_val = torch.tensor(X_seq_val, dtype=torch.float32).to(device)
X_seq_test = torch.tensor(X_seq_test, dtype=torch.float32).to(device)
y_seq_train = torch.tensor(y_seq_train, dtype=torch.float32).to(device)
y_seq_val = torch.tensor(y_seq_val, dtype=torch.float32).to(device)
y_seq_test = torch.tensor(y_seq_test, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_seq_train, y_seq_train)
val_dataset = TensorDataset(X_seq_val, y_seq_val)
test_dataset = TensorDataset(X_seq_test, y_seq_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = KanSleepNet(seq_length=15).to(device)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 1.5, 1, 1, 1]).to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)

num_epochs = 150

best_val_loss = float('inf')
best_val_acc = 0.0
best_epoch = -1

train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

patience = 20
epochs_no_improve = 0
early_stop = False

epoch_true_train = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)

        loss = criterion(outputs.reshape(-1, outputs.size(-1)), y_batch.reshape(-1, y_batch.size(-1)))

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.reshape(-1, outputs.size(-1)), -1)
        _, labels = torch.max(y_batch.reshape(-1, y_batch.size(-1)), -1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.numel()

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch + 1}, Learning Rate: {current_lr}")

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)

            loss = criterion(outputs.reshape(-1, outputs.size(-1)), y_batch.reshape(-1, y_batch.size(-1)))

            val_loss += loss.item()
            _, predicted = torch.max(outputs.reshape(-1, outputs.size(-1)), -1)
            _, labels = torch.max(y_batch.reshape(-1, y_batch.size(-1)), -1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.numel()
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), f'{file_name}model.pth')

        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if val_acc > best_val_acc:
        best_val_acc = val_acc

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    if epochs_no_improve >= patience:
        epoch_true_train = epoch + 1
        print(f'Early stopping at epoch {epoch + 1}')
        early_stop = True
        break
    if early_stop:
        break

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(train_acc_list)
plt.plot(val_acc_list)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(train_loss_list)
plt.plot(val_loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.suptitle('Training History')
plt.savefig(f'{file_name}training_history.png')
plt.close()

# test
model.load_state_dict(torch.load(f'{file_name}model.pth'))
model.eval()
y_seq_pred = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        outputs = model(X_batch)
        y_seq_pred.append(outputs.cpu().numpy())
y_seq_pred = np.concatenate(y_seq_pred, axis=0)

y_seq_pred_ = y_seq_pred.reshape(-1, 5)
y_seq_test_ = y_seq_test.cpu().numpy().reshape(-1, 5)
y_seq_pred_ = np.array([np.argmax(s) for s in y_seq_pred_])
y_seq_test_ = np.array([np.argmax(s) for s in y_seq_test_])

accuracy = accuracy_score(y_seq_test_, y_seq_pred_)
print('Accuracy:', accuracy)

kappa = cohen_kappa_score(y_seq_test_, y_seq_pred_)
print('Kappa:', kappa)

f1_macro = f1_score(y_seq_test_, y_seq_pred_, average='macro')
print('F1 Score (Macro):', f1_macro)
f1_micro = f1_score(y_seq_test_, y_seq_pred_, average='micro')
print('F1 Score (Micro):', f1_micro)
f1_weighted = f1_score(y_seq_test_, y_seq_pred_, average='weighted')
print('F1 Score (Weighted):', f1_weighted)

f1_per_class = f1_score(y_seq_test_, y_seq_pred_, average=None)
print('F1 Score (Per Class):', f1_per_class)

labels = ['Wake', 'N1', 'N2', 'N3', 'REM']

report = classification_report(y_true=y_seq_test_, y_pred=y_seq_pred_, target_names=labels, output_dict=True)
print('Report:', report)
report = pd.DataFrame(report).transpose()
report.to_csv(f'{file_name}report.csv', index=True)

# Confusion matrix with numerical labels
cm = confusion_matrix(y_seq_test_, y_seq_pred_)
sns.heatmap(cm, square=True, annot=True, fmt='d', cmap='coolwarm', xticklabels=labels, yticklabels=labels, cbar=True)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix (Counts)')
plt.savefig(f'{file_name}confusion_matrix.png', bbox_inches='tight', dpi=300)
plt.close()

cm_norm = confusion_matrix(y_seq_test_, y_seq_pred_, normalize='true')
sns.heatmap(cm_norm, square=True, annot=True, fmt='.2f', cmap='coolwarm', xticklabels=labels, yticklabels=labels, cbar=True)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix (Normalized)')
plt.savefig(f'{file_name}confusion_matrix_normalized.png', bbox_inches='tight', dpi=300)
plt.close()

results = {
    'epoch': list(range(1, (epoch_true_train if epoch_true_train else num_epochs) + 1)),  # num_epochs
    'train_loss': train_loss_list,
    'train_acc': train_acc_list,
    'val_loss': val_loss_list,
    'val_acc': val_acc_list
}
results_df = pd.DataFrame(results)
results_df['best_epoch'] = best_epoch
results_df['best_val_acc'] = best_val_acc
results_df.to_csv(f'{file_name}training_results.csv', index=False)

test_results = {
    'accuracy': [accuracy],
    'kappa': [kappa]
}
test_report_df = pd.DataFrame(report).transpose()
test_results_df = pd.DataFrame(test_results)
test_results_df = pd.concat([test_results_df, test_report_df], axis=1)

test_results_df.to_csv(f'{file_name}test_results.csv', index=True)
