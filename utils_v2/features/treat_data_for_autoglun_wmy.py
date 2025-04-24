# used in bayes_opt_ensemble_v2

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def treat_data_autoglun(X, y,label):
    # 处理X
    if isinstance(X, pd.DataFrame):
        print("X 是 Pandas DataFrame")
    else:
        print("X 不是 Pandas DataFrame")
        # 生成默认列名（适用于numpy数组）
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # 确保二维
        num_features = X.shape[1]
        qt_columns = [f'feat_{i}' for i in range(num_features)]
        X = pd.DataFrame(X, columns=qt_columns)
        #X.to_excel('X_temp.xlsx', index=False)  # index=False 不保存索引列
    
    # 处理 y
    if isinstance(y, (pd.DataFrame, pd.Series)):
        print("y 是 Pandas 对象")
        #y = pd.DataFrame(y, columns=[label])  # 处理Series和多维DataFrame
        #y = pd.DataFrame(y.squeeze(), columns=[label])  # 处理Series和多维DataFrame
        #y.columns = [label]  # 统一列名
        #y = pd.DataFrame(y, columns=[label])
    else:
        print("y 不是 Pandas 对象")
        if isinstance(y, np.ndarray) and y.ndim == 1:
            y = y.reshape(-1, 1)  # 确保二维
        y = pd.DataFrame(y, columns=[label])
    
    # 重置索引避免错位
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    # 合并数据
    data = pd.concat([X, y], axis=1)
    return data