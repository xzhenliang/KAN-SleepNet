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

# 设置GPU的索引
gpu_index = 1
device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# input data
data_path = '/data/sleepedf/sleep-cassette/eeg_fpz_cz'

# output file name
file_name = 'KAN_SleepNet_SleepEDF-78_'

fnames = sorted(glob(os.path.join(data_path, '*.npz')))


# Step 1: 提取患者ID并分组
def extract_patient_id(fname):
    """从文件名中提取患者ID，去掉最后的数字和字母"""
    basename = os.path.basename(fname)  # 如 "SC4822G0.npz"
    # 提取患者ID部分：去掉最后的数字和字母（如去掉"2G0"）
    # 规则：找到最后一个数字的位置，然后向前找到第一个字母
    patient_id = basename.split('.')[0]  # 去掉扩展名
    # 去掉最后3个字符（数字+字母+数字），保留患者核心ID
    # 如 SC4822G0 -> SC482, SC4812G0 -> SC481
    if len(patient_id) >= 5:
        # 保留除了最后3个字符之外的部分
        core_id = patient_id[:-3]
        return core_id
    return patient_id  # 如果长度不够，返回原ID

# 按患者ID分组
patient_groups = {}
for fname in fnames:
    patient_id = extract_patient_id(fname)
    if patient_id not in patient_groups:
        patient_groups[patient_id] = []
    patient_groups[patient_id].append(fname)

print(f"Total patients: {len(patient_groups)}")
print("Patient groups:")
for patient_id, files in patient_groups.items():
    print(f"  {patient_id}: {files}")

# Step 2: 按患者组织数据
subjects_X, subjects_y = [], []
patient_ids = []  # 记录每个数据段对应的患者ID

for patient_id, patient_files in patient_groups.items():
    for fname in patient_files:
        samples = np.load(fname)
        x = samples['x']
        y = samples['y']

        # one-hot encoding
        y_oh = np.zeros((len(y), 5))
        for i, label in enumerate(y):
            y_oh[i, label] = 1.

        # 生成序列
        seq_length = 15
        X_seq, y_seq = [], []
        for j in range(0, len(x), seq_length):
            if j + seq_length < len(x):
                X_seq.append(x[j:j + seq_length])
                y_seq.append(y_oh[j:j + seq_length])

        # 将当前文件的所有序列添加到数据中
        for seq_idx in range(len(X_seq)):
            subjects_X.append(np.array(X_seq[seq_idx]))
            subjects_y.append(np.array(y_seq[seq_idx]))
            patient_ids.append(patient_id)  # 记录患者ID

# 转换为numpy数组
subjects_X = np.array(subjects_X)
subjects_y = np.array(subjects_y)
patient_ids = np.array(patient_ids)

print(f"Total sequences: {len(subjects_X)}")
print(f"Unique patients: {len(np.unique(patient_ids))}")

# Step 3: 按患者ID划分数据集
unique_patient_ids = np.unique(patient_ids)

# 首先划分训练集和测试集
train_patients, test_patients = train_test_split(unique_patient_ids, test_size=0.15, random_state=42)

# 再从训练集中划分验证集
train_patients, val_patients = train_test_split(train_patients, test_size=15/85, random_state=42)

print(f"Train patients: {len(train_patients)}")
print(f"Val patients: {len(val_patients)}")
print(f"Test patients: {len(test_patients)}")

# Step 4: 根据患者划分获取数据索引
train_indices = np.where(np.isin(patient_ids, train_patients))[0]
val_indices = np.where(np.isin(patient_ids, val_patients))[0]
test_indices = np.where(np.isin(patient_ids, test_patients))[0]

print(f"Train sequences: {len(train_indices)}")
print(f"Val sequences: {len(val_indices)}")
print(f"Test sequences: {len(test_indices)}")

# Step 5: 获取划分后的数据
X_seq_train = subjects_X[train_indices]
y_seq_train = subjects_y[train_indices]
X_seq_val = subjects_X[val_indices]
y_seq_val = subjects_y[val_indices]
X_seq_test = subjects_X[test_indices]
y_seq_test = subjects_y[test_indices]

# Step 6: 扩展维度
X_seq_train = np.expand_dims(X_seq_train, 2)
X_seq_val = np.expand_dims(X_seq_val, 2)
X_seq_test = np.expand_dims(X_seq_test, 2)

# Step 7: 转换为Tensor
X_seq_train = torch.tensor(X_seq_train, dtype=torch.float32).to(device)
X_seq_val = torch.tensor(X_seq_val, dtype=torch.float32).to(device)
X_seq_test = torch.tensor(X_seq_test, dtype=torch.float32).to(device)
y_seq_train = torch.tensor(y_seq_train, dtype=torch.float32).to(device)
y_seq_val = torch.tensor(y_seq_val, dtype=torch.float32).to(device)
y_seq_test = torch.tensor(y_seq_test, dtype=torch.float32).to(device)

# Step 8: 构建 DataLoader
train_dataset = TensorDataset(X_seq_train, y_seq_train)
val_dataset = TensorDataset(X_seq_val, y_seq_val)
test_dataset = TensorDataset(X_seq_test, y_seq_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# 验证患者划分是否正确
def check_patient_split(patient_ids, split_indices, split_name):
    split_patients = np.unique(patient_ids[split_indices])
    print(f"{split_name} patients: {len(split_patients)}")
    print(f"{split_name} patient IDs: {sorted(split_patients)}")
    return split_patients


train_patients_actual = check_patient_split(patient_ids, train_indices, "Train")
val_patients_actual = check_patient_split(patient_ids, val_indices, "Validation")
test_patients_actual = check_patient_split(patient_ids, test_indices, "Test")

# 检查是否有重叠
train_val_overlap = set(train_patients_actual) & set(val_patients_actual)
train_test_overlap = set(train_patients_actual) & set(test_patients_actual)
val_test_overlap = set(val_patients_actual) & set(test_patients_actual)

print(f"Train-Val overlap: {train_val_overlap}")
print(f"Train-Test overlap: {train_test_overlap}")
print(f"Val-Test overlap: {val_test_overlap}")

if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
    print("✓ Patient splitting is correct - no patient overlap between splits!")
else:
    print("✗ Error: Patient overlap detected between splits!")

# 模型训练
model = KanSleepNet(seq_length=15).to(device)  # seq_length = 15

# 损失函数
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 1.5, 1, 1, 1]).to(device))

# 优化器 指数衰减学习率
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
# 创建学习率指数衰减调度器
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)

num_epochs = 150

best_val_loss = float('inf')
best_val_acc = 0.0
best_epoch = -1

train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

# 提前停止的参数
patience = 20  # 多少个epoch没有改善后停止
epochs_no_improve = 0  # 跟踪没有改善的epoch数
early_stop = False  # 标志是否提前停止

epoch_true_train = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)

        # 使用类不平衡损失函数
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), y_batch.reshape(-1, y_batch.size(-1)))

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.reshape(-1, outputs.size(-1)), -1)
        _, labels = torch.max(y_batch.reshape(-1, y_batch.size(-1)), -1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.numel()

    # 每个 epoch 结束时更新学习率
    scheduler.step()
    # 打印当前学习率
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

            # 使用类不平衡损失函数
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

        # 早停机制，检查验证损失是否改善
        epochs_no_improve = 0  # 重置没有改善的epoch数
    else:
        epochs_no_improve += 1  # 增加没有改善的epoch数

    if val_acc > best_val_acc:
        best_val_acc = val_acc

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    # 检查是否需要提前停止
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

# 测试
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
# 返回每个类别的 F1-score
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
sns.heatmap(cm_norm, square=True, annot=True, fmt='.2f', cmap='coolwarm', xticklabels=labels, yticklabels=labels,
            cbar=True)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix (Normalized)')
plt.savefig(f'{file_name}confusion_matrix_normalized.png', bbox_inches='tight', dpi=300)
plt.close()

# 记录并保存训练和测试结果到 CSV 文件
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

# 记录测试结果到 CSV 文件
test_results = {
    'accuracy': [accuracy],
    'kappa': [kappa]
}
test_report_df = pd.DataFrame(report).transpose()
test_results_df = pd.DataFrame(test_results)
test_results_df = pd.concat([test_results_df, test_report_df], axis=1)
test_results_df.to_csv(f'{file_name}test_results.csv', index=True)