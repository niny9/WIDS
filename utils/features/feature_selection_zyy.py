# for WIDS project.

from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

class FeatureSelection:
    def __init__(self, quantitative_data = [], categorical_data = [],connectome_data = [], solutions = [], min_features = 20):
        self.quantitative_data = quantitative_data
        self.categorical_data = categorical_data
        self.connectome_data = connectome_data
        self.solutions = solutions
        self.min_features = min_features

    def select_quantitative_features(self):
        # 定量特征选择（使用ANOVA F值）
        min_features = self.min_features
        k_quantitative = min(min_features, self.quantitative_data.shape[1])
        if k_quantitative > 0:
            quantitative_selector = SelectKBest(f_classif, k=k_quantitative)
            quantitative_selected = quantitative_selector.fit_transform(
                self.quantitative_data,
                self.solutions.fillna(0.5)
            )
            selected_indices = quantitative_selector.get_support(indices=True)
            print('选中的定量特征索引:', selected_indices)
            """
            下面这段代码失效，应为执行了poly的操作，特证名已经变化
            # 获取选中特征的索引[1,2](@ref)
                selected_indices = quantitative_selector.get_support(indices=True)
                # 从原始特征名列表中提取名称（假设self.quantitative_columns存储特征名）
                quantitatived_features = [self.quantitative_columns[i] for i in selected_indices]
            """
        else:
            quantitative_selected = self.quantitative_data
            """
            quantitatived_features = self.quantitative_columns  # 或返回全部特征名
            """
            
        """
        # 打印选中特征名称
        print(f"Selected quantitative features ({k_quantitative}): {quantitatived_features}")
        """

        return quantitative_selected, quantitative_selector

    
    def select_categorical_features(self):
        #  分类特征选择（使用互信息）
        min_features = self.min_features
        k_categorical = min(min_features, self.categorical_data.shape[1])
        if k_categorical > 0:
            categorical_selector = SelectKBest(mutual_info_classif, k=k_categorical)
            categorical_selected = categorical_selector.fit_transform(
                self.categorical_data,
                self.solutions.fillna(0.5)
            )
            selected_indices = categorical_selector.get_support(indices=True)
            print('选中的分类特征索引:', selected_indices)
        else:
            categorical_selected = self.categorical_data

        return categorical_selected, categorical_selector
    
    
    def select_connectome_features(self):
        # 功能连接数据特征选择（筛选与ADHD相关的主成分）
        min_features = self.min_features
        k_functional = min(min_features, self.connectome_data.shape[1])
        if k_functional > 0:
            functional_selector = SelectKBest(f_classif, k=k_functional)
            functional_selected = functional_selector.fit_transform(
                self.connectome_data,
                self.solutions.fillna(0.5)
            )
            selected_indices = functional_selector.get_support(indices=True)
            print('选中的功能连接数据特征索引:', selected_indices)
        else:
            functional_selected = self.connectome_data
        
        return functional_selected, functional_selector
        
    