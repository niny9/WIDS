# GNN(other all)_adhd（0.81）+sex（0.17）_.1.0_250416.py
import torch
import torch.nn as nn  # 添加这一行
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# 首先定义GCN类
# 修改GCN类
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, metadata_dim,device):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)        
        self.device = device
        # 修改这里，确保使用正确的元数据维度
        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    # 修改GCN类的forward方法
    def forward(self, data):
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print(f"训练时使用设备: {device}")
        # 将数据移动到设备
        device = self.device
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