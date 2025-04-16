# 纯训练一个结果
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
from sklearn.metrics import f1_score  # 添加这一行

import geomstats.datasets.utils as data_utils
import geomstats.backend as gs
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import seaborn as sns

# 修改数据路径为本地路径
base_path = "/Users/niny/Desktop/widsdatathon2025"
train_path = f"{base_path}/TRAIN"
test_path = f"{base_path}/TEST"

# Read in the data
# 修改数据加载部分
df_soln = pd.read_excel(f"{train_path}/TRAINING_SOLUTIONS.xlsx")
df_conn = pd.read_csv(f"{train_path}/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv")

# 修改为提取性别标签
df_soln_sex = df_soln[['participant_id', 'Sex_F']].sort_values('participant_id')
df_conn = df_conn.sort_values('participant_id')

# If needed, load the test data
# test_conn = pd.read_csv(f"{test_path}/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
# test_submission = pd.read_excel(f"{test_path}/TEST_SUBMISSION.xlsx")

# Define the load_connectomes function
def load_connectomes(df_conn, df_soln_sex, as_vectors=False):
    """
    Load brain connectome data and sex labels, returning symmetric matrices with ones on the diagonal.
    """
    
    patient_id = gs.array(df_conn['participant_id'])
    data = gs.array(df_conn.drop('participant_id', axis=1))
    target = gs.array(df_soln_sex['Sex_F'])

    if as_vectors:
        return data, patient_id, target
    mat = SkewSymmetricMatrices(200).matrix_representation(data)
    mat = gs.eye(200) - gs.transpose(gs.tril(mat), (0, 2, 1))
    mat = 1.0 / 2.0 * (mat + gs.transpose(mat, (0, 2, 1)))

    return mat, patient_id, target

# Call the load_connectomes function
data, patient_id, labels = load_connectomes(df_conn, df_soln_sex)

# Print the results
# 修改打印结果
print(f"There are {len(data)} connectomes: {sum(labels==0)} male patients and {sum(labels==1)} female patients.")

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

print("Full dataset class distribution:")
print(pd.Series(y).value_counts(normalize=True) * 100)

print("\nTrain dataset class distribution:")
print(pd.Series(y_train).value_counts(normalize=True) * 100)

print("\nTest dataset class distribution:")
print(pd.Series(y_test).value_counts(normalize=True) * 100)

mdm.fit(X_train, y_train)
print(mdm.score(X_test, y_test))

y_pred = mdm.predict(X_test)
print("F1 score:", f1_score(y_test, y_pred))

# Convert data to torch tensors
connectivity_matrices = torch.tensor(data_corrected).float()
labels = torch.tensor(labels).float()

# Create graph data objects for each matrix
data_list = []
for i in tqdm(range(len(connectivity_matrices)), desc="创建图数据"):
    matrix = connectivity_matrices[i]
    edge_index = (matrix > 0).nonzero(as_tuple=False).t()
    edge_attr = matrix[edge_index[0], edge_index[1]]
    x = torch.eye(200)

    # Create graph data object
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=labels[i].unsqueeze(0))
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
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels=200, out_channels=128)
        self.conv2 = GCNConv(in_channels=128, out_channels=1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        return x.view(-1)

# 实例化模型和其他组件
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# 定义训练函数
def train():
    model.train()
    total_loss = 0
    for data in tqdm(train_loader, desc="Batch训练", leave=False):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 训练循环
losses = []
for epoch in tqdm(range(200), desc="训练进度"):
    loss = train()
    losses.append(loss)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# 评估模型
model.eval()
actual_labels = []
predicted_labels = []
with torch.no_grad():
    for data in tqdm(test_loader, desc="模型评估"):
        out = model(data)
        pred = torch.round(torch.sigmoid(out))
        actual_labels.extend(data.y.int().tolist())
        predicted_labels.extend(pred.tolist())

accuracy = sum(1 for a, p in zip(actual_labels, predicted_labels) if a == p) / len(actual_labels)
print(f'Accuracy: {accuracy:.4f}')

epochs = list(range(1, len(losses) + 1))

# Visualize the loss values
plt.figure(figsize=(6, 3))
plt.plot(epochs, losses, marker='o', linestyle='-', color='b')
plt.title('Training loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()



# Confusion matrix

true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

for true, predicted in zip(actual_labels, predicted_labels):
    if predicted == 1 and true == 1:
        true_positives += 1
    elif predicted == 1 and true == 0:
        false_positives += 1
    elif predicted == 0 and true == 0:
        true_negatives += 1
    elif predicted == 0 and true == 1:
        false_negatives += 1

conf_matrix = np.array([[true_positives, false_negatives],
                        [false_positives, true_negatives]])

cm_labels = ['Positive', 'Negative']
categories = ['Positive', 'Negative']

plt.figure(figsize=(6,3))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=categories)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score
f1 = f1_score(actual_labels, predicted_labels)
print("F1 score:", f1)

# 修改类别分布的显示
print("性别类别分布:")
print("完整数据集:", pd.Series(y).value_counts(normalize=True) * 100)
print("训练集:", pd.Series(y_train).value_counts(normalize=True) * 100)
print("测试集:", pd.Series(y_test).value_counts(normalize=True) * 100)

# 修改混淆矩阵的标签
cm_labels = ['Female', 'Male']
categories = ['Female', 'Male']

plt.figure(figsize=(6,3))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=categories)
plt.title('Gender Prediction Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# 修改最终评估指标的显示
print("\n性别预测结果:")
print("准确率:", accuracy)
print("F1分数:", f1)

