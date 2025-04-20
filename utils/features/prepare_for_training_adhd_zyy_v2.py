"""
相比原版，修改如下：
1. 增加test数据
2. 调整了结构
quantitative_metadata -->train_quantitative / test_quantitative
categorical_metadata -->train_cat / test_cat
connectome_metadata -->train_connectome / test_connectome

modified at 20250411
解决训练特征和测试特征不一致的问题
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('../utils')
from features.data_preprocessing_zyy import DataPreprocessing
from features.feature_selection_zyy import FeatureSelection
from imblearn.over_sampling import SMOTE

def prepare_for_training(train_connectome, train_quantitative, train_cat,test_connectome, test_quantitative, test_cat,train_solutions, test_solutions):

    """ 数据预处理 """
    dp_train = DataPreprocessing(quantitative_data = train_quantitative, categorical_data = train_cat,
                                 connectome_data = train_connectome)
    dp_test = DataPreprocessing(quantitative_data = test_quantitative, categorical_data = test_cat, 
                                connectome_data = test_connectome)

    # 1.1 定量数据处理
    train_quantitative_treat = dp_train.treat_quantitative_data()
    test_quantitative_treat = dp_test.treat_quantitative_data()

    # 1.2 分类数据处理
    train_cat_treat = dp_train.treat_categorical_data()
    test_cat_treat = dp_test.treat_categorical_data()

    # 1.3 功能连接数据处理
    train_connectome_treat = dp_train.treat_connectome_data(reduce_dim_method='PCA')
    test_connectome_treat = dp_test.treat_connectome_data(reduce_dim_method='PCA')

    """ 2. 特征选择 """
    feature_select_train = FeatureSelection(quantitative_data = train_quantitative_treat, categorical_data = train_cat_treat,
                                             connectome_data = train_connectome_treat, solutions = train_solutions)

    """
    feature_select_test = FeatureSelection(quantitative_data = test_quantitative_treat, categorical_data = test_cat_treat, 
                                           connectome_data = test_connectome_treat, solutions = test_solutions)
    """
    
    # 2.1 定量特征选择
    print('Selecting quantitative features for training data...')
    quantitative_train_selected,quantitative_selector = feature_select_train.select_quantitative_features()
    print('Number of quantitative faetures for training data: ', quantitative_train_selected.shape[1])
    print('Selecting quantitative features for testing data...')
    quantitative_test_selected = quantitative_selector.fit_transform(
                                    test_quantitative_treat,
                                    test_solutions.fillna(0.5)
                                )
    #quantitative_test_selected = test_quantitative_treat.columns[quan_feture_indices]
    print('Number ofquantitative faetures for testing data: ', quantitative_test_selected.shape[1])

    # 2.2 分类特征选择
    print('Selecting categorical features for training data...')
    cat_train_selected,cat_selector = feature_select_train.select_categorical_features()
    print('Number of categorical faetures for training data: ', cat_train_selected.shape[1])
    print('Selecting categorical features for testing data...')
    #cat_test_selected = feature_select_test.select_categorical_features()
    cat_test_selected = cat_selector.fit_transform(
                                    test_cat_treat,
                                    test_solutions.fillna(0.5)
                                )
    #cat_test_selected = test_cat_treat.columns[cat_feture_indices]
    print('Number of categorical faetures for testing data: ', cat_test_selected.shape[1])

    # 2.3 功能连接特征选择
    print('Selecting connectome features for training data...')
    connectome_train_selected,connectome_selector = feature_select_train.select_connectome_features()
    print('Number of connectome faetures for training data: ', connectome_train_selected.shape[1])
    print('Selecting connectome features for testing data...')
    #connectome_test_selected = feature_select_test.select_connectome_features()
    connectome_test_selected = connectome_selector.fit_transform(
                                    test_connectome_treat,
                                    test_solutions.fillna(0.5)
                                )  
    print('Number of connectome faetures for testing data: ', connectome_test_selected.shape[1])

    """ 3. 数据增强 """
    """
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(
        np.hstack([quantitative_train_selected, cat_train_selected, connectome_train_selected]),
        train_solutions.fillna(0.5).astype(int)
    )
    """
    
    X_train = np.hstack([quantitative_train_selected, cat_train_selected, connectome_train_selected])
    y_train = train_solutions
    """ 4. 构建测试集 """
    #X_test = pd.merge(quantitative_test_selected, cat_test_selected, connectome_test_selected, on = 'participant_id')
    X_test = np.hstack([quantitative_test_selected, cat_test_selected, connectome_test_selected])
    y_test = test_solutions
    
    return X_train, y_train, X_test, y_test
    