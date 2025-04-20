# for WIDS project.

import pandas as pd
import numpy as np
import umap
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
import sys
sys.path.append('../utils')
from features.connectome_analysis import connect_analysis

class DataPreprocessing:
    def __init__(self, quantitative_data = [], categorical_data = [],connectome_data = []):
        self.quantitative_data = quantitative_data
        self.categorical_data = categorical_data
        self.connectome_data = connectome_data

    def treat_quantitative_data(self):
        """
        处理定量数据：
        1. 删除非数值列（如participant_id）
        2. 将非数值数据转换为NaN
        3. 用均值填充缺失值
        """
        # 只保留数值列
        df = self.quantitative_data.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[numeric_cols]

        # 将非数值数据转换为NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 用均值填充缺失值
        imputer = SimpleImputer(strategy='mean')
        df_imputed = imputer.fit_transform(df)
        df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

        # 标准化 + 添加多项式
        quantitative_pipe = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False)  # 添加多项式特征
        )
        quantitative_processed = quantitative_pipe.fit_transform(df_imputed)
        return quantitative_processed

    def treat_categorical_data(self):
        """
        处理定性数据：
        1. 删除非数值列（如participant_id） + 数值列（如MRI_Track_Age_at_Scan）
        2. 将非数值数据转换为NaN
        3. 用One-Hot编码
        3. 用众数填充缺失值
        """
        df = self.categorical_data.copy()
        categorical_columns = df.select_dtypes(include=['object']).columns
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        categorical_encoded = encoder.fit_transform(df[categorical_columns])
        categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_columns))
        return categorical_encoded_df

    def treat_connectome_data(self,reduce_dim_method='PCA'):
        """
        处理功能连接数据： 
        1. 删除participant_id列
        2. 用PCA降维
        """
        df = self.connectome_data.copy()
        functional_data = df.iloc[:, 1:]  # 排除 participant_id
        #print("列名类型示例:", (df.columns[0]))  # 应输出 <class 'str'>
        count_connect_contra,count_connect_within = connect_analysis(functional_data, normalized=1)
        contra_df = pd.DataFrame(count_connect_contra, columns=['contra_feature'])  # 确保列名为字符串
        within_df = pd.DataFrame(count_connect_within, columns=['within_feature'])
        #print("列名类型示例:", (functional_data.columns[0]))  # 应输出 <class 'str'>
        """
        # 打印数据信息
        print("functional_data shape:", functional_data.shape)
        print("count_contra_2d shape:", count_connect_contra.shape)
        print("count_within_2d shape:", count_connect_within.shape)
        print(type(functional_data))
        print(type(count_connect_contra))
        print(type(count_connect_within))
        """
        functional_data = pd.concat([functional_data, contra_df, within_df], axis=1)  # panda 类型
        # functional_data = np.hstack([functional_data, count_connect_contra, count_connect_within])   # numpy类型
        # print("Final functional_data shape:", functional_data.shape)
        #print("列名类型示例:", type(functional_data.columns[0]))  # 应输出 <class 'str'>
        #print("列名类型检查:", set(map(type, functional_data.columns)))
         # 强制所有列名为字符串
        functional_data.columns = functional_data.columns.astype(str)
        #print("列名类型检查:", set(map(type, functional_data.columns)))  # 确认无混合类型
        if reduce_dim_method == 'PCA':
            # PCA降维
            pca = PCA(n_components=50)
            functional_pca = pca.fit_transform(functional_data)
            functional_reduced_df = pd.DataFrame(functional_pca, columns=[f'PCA_{i}' for i in range(50)])
        elif reduce_dim_method == 'UMAP':
            # UMAP 降维
            umap_pipe = make_pipeline(SimpleImputer(), MinMaxScaler())
            umap_reducer = umap.UMAP()
            functional_umap = umap_reducer.fit_transform(umap_pipe.fit_transform(functional_data))
            functional_reduced_df = pd.DataFrame(functional_umap, columns=[f'UMAP_{i}' for i in range(functional_umap.shape[1])])

        return functional_reduced_df