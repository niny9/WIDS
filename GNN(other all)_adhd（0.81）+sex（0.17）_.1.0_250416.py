# 训练2个预测结果+增加每个患者的其他特征
# Geomstats is an open-source Python package for computations and statistics on nonlinear manifolds
# pip install geomstats
# pip install openpyxl
# pip install torch_geometric
# pip install tqdm

# 在导入部分添加
import numpy as np
import pandas as pd
import csv
import openpyxl
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix  # 修改这行
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import geomstats.datasets.utils as data_utils
import geomstats.backend as gs
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices

import torch
import torch.nn as nn  # 添加这一行
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import seaborn as sns

# 添加特征处理类（在数据加载之前）
class FeatureProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
    def fit_transform(self, metadata_df, numeric_cols, categorical_cols):
        # 数值特征
        numeric_features = self.scaler.fit_transform(metadata_df[numeric_cols])
        
        # 类别特征
        metadata_df[categorical_cols] = metadata_df[categorical_cols].astype(str)
        categorical_features = self.encoder.fit_transform(metadata_df[categorical_cols])
        
        return np.concatenate([numeric_features, categorical_features], axis=1)
    
    def transform(self, metadata_df, numeric_cols, categorical_cols):
        # 数值特征
        numeric_features = self.scaler.transform(metadata_df[numeric_cols])
        
        # 类别特征
        metadata_df[categorical_cols] = metadata_df[categorical_cols].astype(str)
        categorical_features = self.encoder.transform(metadata_df[categorical_cols])
        
        return np.concatenate([numeric_features, categorical_features], axis=1)

# 修改数据路径为本地路径
base_path = "/Users/niny/Desktop/widsdatathon2025"
train_path = f"{base_path}/TRAIN"
test_path = f"{base_path}/TEST"

# Read in the data
df_soln = pd.read_excel(f"{train_path}/TRAINING_SOLUTIONS.xlsx")
df_conn = pd.read_csv(f"{train_path}/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv")
# 添加元数据读取
train_metadata = pd.read_excel(f"{train_path}/TRAIN_CATEGORICAL_METADATA_new.xlsx")
test_metadata = pd.read_excel(f"{test_path}/TEST_CATEGORICAL.xlsx")

# 定义要使用的特征列
numeric_cols = ['Basic_Demos_Enroll_Year', 
               'Barratt_Barratt_P1_Edu', 'Barratt_Barratt_P1_Occ',
               'Barratt_Barratt_P2_Edu', 'Barratt_Barratt_P2_Occ']

categorical_cols = ['Basic_Demos_Study_Site',
                   'PreInt_Demos_Fam_Child_Ethnicity',
                   'PreInt_Demos_Fam_Child_Race',
                   'MRI_Track_Scan_Location']

# 修改数据加载部分
# Extract both ADHD and Sex labels
df_soln = df_soln[['participant_id', 'ADHD_Outcome', 'Sex_F']].sort_values('participant_id')
df_conn = df_conn.sort_values('participant_id')
train_metadata = train_metadata.sort_values('participant_id')  # 确保元数据也按ID排序

# 确认三个数据集的ID是否匹配
print("连接组数据集ID数量:", len(df_conn['participant_id'].unique()))
print("标签数据集ID数量:", len(df_soln['participant_id'].unique()))
print("元数据数据集ID数量:", len(train_metadata['participant_id'].unique()))

# 检查ID是否完全匹配
conn_ids = set(df_conn['participant_id'])
soln_ids = set(df_soln['participant_id'])
meta_ids = set(train_metadata['participant_id'])
print("所有ID都匹配:", conn_ids == soln_ids == meta_ids)

# 如果ID不匹配，找出共同的ID
common_ids = conn_ids.intersection(soln_ids).intersection(meta_ids)
print("共同ID数量:", len(common_ids))

# 只使用共同的ID
df_conn = df_conn[df_conn['participant_id'].isin(common_ids)]
df_soln = df_soln[df_soln['participant_id'].isin(common_ids)]
train_metadata = train_metadata[train_metadata['participant_id'].isin(common_ids)]

# 定义load_connectomes函数
def load_connectomes(df_conn, df_soln, as_vectors=False):
    patient_id = gs.array(df_conn['participant_id'])
    data = gs.array(df_conn.drop('participant_id', axis=1))
    target_adhd = gs.array(df_soln['ADHD_Outcome'])
    target_sex = gs.array(df_soln['Sex_F'])
    # 将两个标签组合成一个二维数组
    targets = np.stack([target_adhd, target_sex], axis=1)

    if as_vectors:
        return data, patient_id, targets
    mat = SkewSymmetricMatrices(200).matrix_representation(data)
    mat = gs.eye(200) - gs.transpose(gs.tril(mat), (0, 2, 1))
    mat = 1.0 / 2.0 * (mat + gs.transpose(mat, (0, 2, 1)))

    return mat, patient_id, targets

# 调用load_connectomes函数
data, patient_id, labels = load_connectomes(df_conn, df_soln)

# Print the results
print(f"There are {len(data)} connectomes: {sum(labels[:,0]==0)} healthy patients and {sum(labels[:,0]==1)} ADHD patients.")
print(f"There are {sum(labels[:,1]==0)} male patients and {sum(labels[:,1]==1)} female patients.")

# If needed, load the test data
# test_conn = pd.read_csv(f"{test_path}/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
# test_submission = pd.read_excel(f"{test_path}/TEST_SUBMISSION.xlsx")

data.shape

from geomstats.geometry.spd_matrices import SPDMatrices

manifold = SPDMatrices(200, equip=False)
print(gs.all(manifold.belongs(data)))

# Count the number of connectomes that do not lie on the SPD manifold

count_false = np.sum(~(manifold.belongs(data)))
print("Count of False:", count_false)
# Function to add a diagonal matrix to a 2D matrix
def add_diagonal_correction(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    min_eigenvalue = np.min(eigenvalues)

    if min_eigenvalue < 0:
        correction = -min_eigenvalue + 1e-6
        correction_matrix = correction * np.eye(matrix.shape[0])
        return matrix + correction_matrix
    else:
        return matrix

# Apply the correction to each 2D slice of the 3D matrix
data_corrected = np.array([add_diagonal_correction(slice) for slice in tqdm(data, desc="矩阵校正")])

print("Original Matrix shape:", data.shape)
print("Corrected Matrix shape:", data_corrected.shape)

print(gs.all(manifold.belongs(data_corrected)))

def count_differences(array1, array2, tolerance=1e-6):
    """
    This function compares two 3D arrays and returns the count of differences.
    """
    if array1.shape != array2.shape:
        raise ValueError("Arrays must be of the same shape")
    
    differences = np.greater(np.abs(array1 - array2), tolerance)
    count = np.sum(differences)
    
    return count

print(count_differences(data, data_corrected))

from geomstats.learning.mdm import RiemannianMinimumDistanceToMean

spd_manifold = SPDMatrices(n=200, equip=True)
mdm = RiemannianMinimumDistanceToMean(space=spd_manifold)

from sklearn.model_selection import train_test_split
X = data_corrected; y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=47)

print(f"The dataset has {len(X)} connectomes.")
print(f"The train set has {len(X_train)} connectomes and has size {X_train.shape}.")
print(f"The test set has {len(X_test)} connectomes and has size {X_test.shape}.")

# 修改类别分布的显示方式
print("\nADHD类别分布:")
print("训练集:", pd.Series(y_train[:,0]).value_counts(normalize=True) * 100)
print("测试集:", pd.Series(y_test[:,0]).value_counts(normalize=True) * 100)

print("\n性别类别分布:")
print("训练集:", pd.Series(y_train[:,1]).value_counts(normalize=True) * 100)
print("测试集:", pd.Series(y_test[:,1]).value_counts(normalize=True) * 100)

# 直接进入神经网络部分，跳过MDM
# Convert data to torch tensors
connectivity_matrices = torch.tensor(data_corrected).float()
labels = torch.tensor(labels).float()  # labels现在包含ADHD和性别两个标签

# 创建图数据对象
# 处理元数据特征
feature_processor = FeatureProcessor()
train_metadata_features = feature_processor.fit_transform(
    train_metadata, numeric_cols, categorical_cols
)
train_metadata_tensor = torch.tensor(train_metadata_features, dtype=torch.float)

# 打印元数据特征的维度，以便确认
print(f"元数据特征维度: {train_metadata_features.shape}")

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 修改创建图数据的部分
data_list = []
for i in tqdm(range(len(connectivity_matrices)), desc="创建图数据"):
    matrix = connectivity_matrices[i]
    edge_index = (matrix > 0).nonzero(as_tuple=False).t()
    edge_attr = matrix[edge_index[0], edge_index[1]]
    x = torch.eye(200)
    
    # 确保元数据索引与连接组矩阵索引一致
    patient_id_i = patient_id[i]
    metadata_idx = train_metadata[train_metadata['participant_id'] == patient_id_i].index[0]
    metadata = train_metadata_tensor[metadata_idx]
    
    # 创建图数据对象 - 不要将元数据移动到设备
    graph_data = Data(x=x, 
                     edge_index=edge_index, 
                     edge_attr=edge_attr,
                     metadata=metadata,  # 不要移动到设备
                     y=labels[i])
    data_list.append(graph_data)

    torch.manual_seed(192024)

# Apply a 80:20 train test split
split_index = int(len(data_list) * 0.75)

# Manually split the data_list into train and test sets
train_data = data_list[:split_index]
test_data = data_list[split_index:]

print("Number of examples in training data:", len(train_data))
print("Number of examples in test data:", len(test_data))

# 首先定义GCN类
# 修改GCN类
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, metadata_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)        
        
        # 修改这里，确保使用正确的元数据维度
        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    # 修改GCN类的forward方法
    def forward(self, data):
        # 将数据移动到设备
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)
        batch = data.batch.to(device)
        
        # GCN处理图结构
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        
        # 处理元数据 - 确保维度正确
        batch_size = x.size(0)
        
        # 完全重写元数据处理逻辑
        # 创建一个新的元数据张量，大小为 [batch_size, metadata_dim]
        metadata_dim = 30  # 从train_metadata_features.shape[1]获取
        batched_metadata = []
        
        # 获取每个图对应的元数据
        for i in range(batch_size):
            # 找到批次中第i个图的节点索引
            node_indices = (batch == i).nonzero(as_tuple=True)[0]
            if len(node_indices) > 0:
                # 获取第一个节点的索引
                node_idx = node_indices[0].item()
                # 获取该节点对应的元数据
                if hasattr(data, 'metadata'):
                    if data.metadata.dim() == 1:
                        # 如果元数据是一维的，整个作为一个样本
                        sample_metadata = data.metadata.to(device)
                    else:
                        # 如果元数据是多维的，获取对应的行
                        # 这里使用模运算确保索引不会越界
                        idx = node_idx % data.metadata.size(0)
                        sample_metadata = data.metadata[idx].to(device)
                    
                    # 确保元数据维度正确
                    if sample_metadata.dim() == 1 and sample_metadata.size(0) == metadata_dim:
                        batched_metadata.append(sample_metadata)
                    else:
                        # 如果维度不正确，使用零张量
                        batched_metadata.append(torch.zeros(metadata_dim, device=device))
                else:
                    # 如果没有元数据，使用零张量
                    batched_metadata.append(torch.zeros(metadata_dim, device=device))
        
        # 将所有元数据堆叠成一个批次
        metadata = torch.stack(batched_metadata)
        
        # 应用MLP
        metadata = self.metadata_mlp(metadata)  # [batch_size, hidden_dim]
        
        # 特征融合
        x = torch.cat([x, metadata], dim=1)  # [batch_size, hidden_dim * 2]
        x = self.fc(x)
        return x

# 实例化模型和其他组件
# 实例化模型时添加元数据维度 - 确保使用正确的维度
metadata_dim = train_metadata_features.shape[1]  # 这应该是960，而不是30
print(f"使用的元数据维度: {metadata_dim}")

model = GCN(
    input_dim=200,
    hidden_dim=128,
    output_dim=2,
    metadata_dim=metadata_dim
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

# 获取实际样本数量
n_samples = len(data_list)
print(f"总样本数量: {n_samples}")

# 修改DataLoader的批次大小
train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
test_loader = DataLoader(test_data, batch_size=len(test_data))

# 训练循环
losses = []
for epoch in tqdm(range(200), desc="训练进度"):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        # 修改：确保data.y的维度正确 [batch_size, 2]
        target = data.y.view(-1, 2).float()  # 重塑维度为 [batch_size, 2]
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    losses.append(total_loss / len(train_loader))
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {losses[-1]:.4f}')

# 评估模型时也需要修改
model.eval()
actual_adhd = []
actual_sex = []
pred_adhd_probs = []
pred_sex_probs = []

with torch.no_grad():
    for data in tqdm(test_loader, desc="模型评估"):
        out = model(data)
        probs = torch.sigmoid(out)  # 获取概率值
        target = data.y.view(-1, 2)  # 同样需要重塑维度
        
        # 分别存储ADHD和性别的真实值和预测概率
        actual_adhd.extend(target[:, 0].int().tolist())
        actual_sex.extend(target[:, 1].int().tolist())
        pred_adhd_probs.extend(probs[:, 0].tolist())
        pred_sex_probs.extend(probs[:, 1].tolist())

# 计算并显示两个任务的指标
pred_adhd_labels = [1 if p >= 0.5 else 0 for p in pred_adhd_probs]
pred_sex_labels = [1 if p >= 0.5 else 0 for p in pred_sex_probs]

print("\nADHD预测结果:")
print("准确率:", accuracy_score(actual_adhd, pred_adhd_labels))
print("F1分数:", f1_score(actual_adhd, pred_adhd_labels))

print("\n性别预测结果:")
print("准确率:", accuracy_score(actual_sex, pred_sex_labels))
print("F1分数:", f1_score(actual_sex, pred_sex_labels))

# 绘制两个预测任务的混淆矩阵
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# ADHD混淆矩阵
conf_matrix_adhd = confusion_matrix(actual_adhd, pred_adhd_labels)
sns.heatmap(conf_matrix_adhd, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('ADHD预测混淆矩阵')
ax1.set_xlabel('预测标签')
ax1.set_ylabel('真实标签')

# 性别混淆矩阵
conf_matrix_sex = confusion_matrix(actual_sex, pred_sex_labels)
sns.heatmap(conf_matrix_sex, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_title('性别预测混淆矩阵')
ax2.set_xlabel('预测标签')
ax2.set_ylabel('真实标签')

plt.tight_layout()
plt.show()

# 保存预测结果
results_df = pd.DataFrame({
    'participant_id': patient_id[len(train_data):],  # 测试集的patient_id
    'ADHD_Outcome_prob': [f"{p*100:.2f}%" for p in pred_adhd_probs],  # 转换为百分比格式
    'Sex_F_prob': [f"{p*100:.2f}%" for p in pred_sex_probs]  # 转换为百分比格式
})
results_df.to_csv('/Users/niny/Downloads/编程学习/Kaggle/prediction_probabilities.csv', index=False)

# 修改模型训练部分，添加学习率调度和早停
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 使用较小的学习率
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# 添加早停机制
best_loss = float('inf')
patience = 20
patience_counter = 0

# 训练循环
losses = []
for epoch in tqdm(range(500), desc="训练进度"):  # 增加训练轮数
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        target = data.y.view(-1, 2).float()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    
    # 更新学习率
    scheduler.step(avg_loss)
    
    # 早停检查
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        # 保存最佳模型
        torch.save(model.state_dict(), '/Users/niny/Downloads/编程学习/Kaggle/best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'早停于第 {epoch} 轮，最佳损失: {best_loss:.4f}')
            break
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

# 加载最佳模型进行评估
model.load_state_dict(torch.load('/Users/niny/Downloads/编程学习/Kaggle/best_model.pt'))

# 评估模型时也需要修改
model.eval()
actual_adhd = []
actual_sex = []
pred_adhd_probs = []
pred_sex_probs = []

with torch.no_grad():
    for data in tqdm(test_loader, desc="模型评估"):
        out = model(data)
        probs = torch.sigmoid(out)  # 获取概率值
        target = data.y.view(-1, 2)  # 同样需要重塑维度
        
        # 分别存储ADHD和性别的真实值和预测概率
        actual_adhd.extend(target[:, 0].int().tolist())
        actual_sex.extend(target[:, 1].int().tolist())
        pred_adhd_probs.extend(probs[:, 0].tolist())
        pred_sex_probs.extend(probs[:, 1].tolist())

# 计算并显示两个任务的指标
pred_adhd_labels = [1 if p >= 0.5 else 0 for p in pred_adhd_probs]
pred_sex_labels = [1 if p >= 0.5 else 0 for p in pred_sex_probs]

print("\nADHD预测结果:")
print("准确率:", accuracy_score(actual_adhd, pred_adhd_labels))
print("F1分数:", f1_score(actual_adhd, pred_adhd_labels))

print("\n性别预测结果:")
print("准确率:", accuracy_score(actual_sex, pred_sex_labels))
print("F1分数:", f1_score(actual_sex, pred_sex_labels))

# 绘制两个预测任务的混淆矩阵
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# ADHD混淆矩阵
conf_matrix_adhd = confusion_matrix(actual_adhd, pred_adhd_labels)
sns.heatmap(conf_matrix_adhd, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('ADHD预测混淆矩阵')
ax1.set_xlabel('预测标签')
ax1.set_ylabel('真实标签')

# 性别混淆矩阵
conf_matrix_sex = confusion_matrix(actual_sex, pred_sex_labels)
sns.heatmap(conf_matrix_sex, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_title('性别预测混淆矩阵')
ax2.set_xlabel('预测标签')
ax2.set_ylabel('真实标签')

plt.tight_layout()
plt.show()

# 保存预测结果
results_df = pd.DataFrame({
    'participant_id': patient_id[len(train_data):],  # 测试集的patient_id
    'ADHD_Outcome_prob': [f"{p*100:.2f}%" for p in pred_adhd_probs],  # 转换为百分比格式
    'Sex_F_prob': [f"{p*100:.2f}%" for p in pred_sex_probs]  # 转换为百分比格式
})
results_df.to_csv('/Users/niny/Downloads/编程学习/Kaggle/prediction_probabilities.csv', index=False)

# 修改批次大小，使用较小的批次
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# 使用较小的学习率
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 添加权重衰减

