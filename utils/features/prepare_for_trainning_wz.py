# GNN(other all)_adhd（0.81）+sex（0.17）_.1.0_250416.py

# 在导入部分添加
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix  # 修改这行
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import geomstats.backend as gs
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices
import torch
from torch_geometric.data import Data

def prepare_for_trainning(train_solutions, train_connectome, train_cat,test_solutions,test_connectome, test_cat, goal):
    # goal : 预测目标，ADHD or sex
    """ 函数1： 处理功能连接矩阵"""
    def treat_connectome(train_solutions,train_connectome,train_metadata):
        # 初始化
        df_soln = train_solutions
        df_conn = train_connectome
        
        """ 数据预处理 """
        # 修改数据加载部分
        # Extract both ADHD and Sex labels
        df_soln = df_soln[['participant_id',goal]].sort_values('participant_id')
        df_conn = df_conn.sort_values('participant_id')

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

        """ 构建功能连接矩阵 """
        def load_connectomes(df_conn, df_soln, as_vectors=False):
            # 加载基础数据（强制float32）
            patient_id = gs.array(df_conn['participant_id'])
            data = gs.array(df_conn.drop('participant_id', axis=1), dtype=np.float32)
            target = gs.array(df_soln.drop('participant_id', axis=1), dtype=np.float32)

            if as_vectors:
                return data, patient_id, target
            
            # 分块生成矩阵
            batch_size = 500
            n_samples = data.shape[0]
            mat = np.zeros((n_samples, 200, 200), dtype=np.float32)
            
            for i in range(0, n_samples, batch_size):
                batch = data[i:i+batch_size]
                mat_batch = SkewSymmetricMatrices(200).matrix_representation(batch)
                mat_batch = gs.eye(200) - gs.transpose(gs.tril(mat_batch), (0, 2, 1))
                mat[i:i+batch_size] = 0.5 * (mat_batch + gs.transpose(mat_batch, (0, 2, 1)))
                del batch, mat_batch

            del data  # 释放原始数据内存
            return mat, patient_id, target
        # 调用load_connectomes函数
        data, patient_id, labels = load_connectomes(df_conn, df_soln)

        # Print the results
        if goal == 'ADHD_Outcome':
            print(f"There are {len(data)} connectomes: {sum(labels[:,0]==0)} healthy patients and {sum(labels[:,0]==1)} ADHD patients.")
        elif goal == 'Sex_F':
            print(f"There are {sum(labels[:,0]==0)} male patients and {sum(labels[:,0]==1)} female patients.")

        print(f"The connectomes have shape {data.shape} and the labels have shape {labels.shape}.")

        from geomstats.geometry.spd_matrices import SPDMatrices

        manifold = SPDMatrices(200, equip=False)
        print(gs.all(manifold.belongs(data)))

        # Count the number of connectomes that do not lie on the SPD manifold

        count_false = np.sum(~(manifold.belongs(data)))
        print("Count of False:", count_false)
        """ 矩阵校正 """
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
        
        return data_corrected, data, patient_id, labels
    
    """ 函数2： 功能连接矩阵对比 """
    def count_differences(array1, array2, tolerance=1e-6):
        """
        This function compares two 3D arrays and returns the count of differences.
        """
        if array1.shape != array2.shape:
            raise ValueError("Arrays must be of the same shape")
        
        differences = np.greater(np.abs(array1 - array2), tolerance)
        count = np.sum(differences)
        
        return count

    """ 函数3： 特征处理 """
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
        
    """ 函数4： 元数据和矩阵合并，构建datalist """
    def construct_datalist(connectivity_matrices, patient_id, labels, train_metadata,train_metadata_tensor):
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


    """ 数据处理主体流程 """    
    # 添加元数据读取
    train_metadata = train_cat
    test_metadata =test_cat

    train_data_corrected, train_data, train_patient_id, train_labels = treat_connectome(train_solutions,train_connectome,train_metadata)
    test_data_corrected, test_data, test_patient_id, test_labels = treat_connectome(test_solutions,test_connectome,test_metadata)
    
    print('train_connectome 原始数据和校正后数据对比：',count_differences(train_data, train_data_corrected))
    print('test_connectome 原始数据和校正后数据对比：',count_differences(test_data, test_data_corrected))
    print('train_labels 和 test labels 的对比：',count_differences(train_labels, test_labels))
    print('train ids 和 test ids 的对比：',count_differences(train_patient_id, test_patient_id))
   
    train_metadata = train_metadata.sort_values('train_patient_id')  # 确保元数据也按ID排序
    test_metadata = test_metadata.sort_values('test_patient_id')  # 确保元数据也按ID排序

    print("Original Matrix shape:", train_data.shape)
    print("Corrected Matrix shape:", train_data_corrected.shape)
    # print(gs.all(manifold.belongs(data_corrected)))
    

    # 直接进入神经网络部分，跳过MDM
    # Convert data to torch tensors
    train_matrices = torch.tensor(train_data_corrected).float()
    test_matrices = torch.tensor(test_data_corrected).float()

    train_labels = torch.tensor(train_labels).float()  
    test_labels = torch.tensor(test_labels).float()

    # 定义要使用的特征列
    numeric_cols = ['Basic_Demos_Enroll_Year', 
                'Barratt_Barratt_P1_Edu', 'Barratt_Barratt_P1_Occ',
                'Barratt_Barratt_P2_Edu', 'Barratt_Barratt_P2_Occ']

    categorical_cols = ['Basic_Demos_Study_Site',
                    'PreInt_Demos_Fam_Child_Ethnicity',
                    'PreInt_Demos_Fam_Child_Race',
                    'MRI_Track_Scan_Location']
    # 创建图数据对象
    # 处理元数据特征
    feature_processor = FeatureProcessor()
    train_metadata_features = feature_processor.fit_transform(
        train_metadata, numeric_cols, categorical_cols
    )
    train_metadata_tensor = torch.tensor(train_metadata_features, dtype=torch.float)

    test_metadata_features = feature_processor.transform(
        test_metadata, numeric_cols, categorical_cols
    )
    test_metadata_tensor = torch.tensor(test_metadata_features, dtype=torch.float)

    # 打印元数据特征的维度，以便确认
    print(f"元数据特征维度: {train_metadata_features.shape}")

    # 修改创建图数据的部分
    train_data = construct_datalist(train_matrices, train_patient_id, train_labels, train_metadata,train_metadata_tensor)
    test_data = construct_datalist(test_matrices, test_patient_id, test_labels, test_metadata,test_metadata_tensor)

    print("Number of examples in training data:", len(train_data))
    print("Number of examples in test data:", len(test_data))
    
    return train_data, test_data, train_metadata_features
