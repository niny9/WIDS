# By zhangting 2025/04/20
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import torch
from torch_geometric.loader import DataLoader
import sys
sys.path.append('../utils')
from features.treat_data_for_autoglun_wmy import treat_data_autoglun
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from autogluon.tabular import TabularPredictor
from GCN_wz import GCN


# 定义集成学习网络
class EnsembleModel():
    def __init__(self,xgb_params,rf_params,hyperparameters, hyperparameter_tune_kwargs,metadata_dim,label):
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        # 模型1特定层
        self.hyperparameters = hyperparameters
        self.hyperparameter_tune_kwargs = hyperparameter_tune_kwargs
        self.label = label
        self.model1 = XGBClassifier(**xgb_params)
        # 模型2特定层
        self.model2 = RandomForestClassifier(**rf_params)
        # 模型2特定层
        self.model3 = TabularPredictor(label=label, eval_metric='f1_weighted')
        # 模型3特定层
        self.model4 = GCN(
                        input_dim=200,
                        hidden_dim=128,
                        output_dim=1,
                        metadata_dim=metadata_dim
                    ).to(device)
        
        # ensemble层
        #self.meta =  LogisticRegression()
        #self.meta =  GradientBoostingClassifier()
        self.meta = VotingClassifier(estimators=[
            ('lr', LogisticRegression()),
            ('gb', GradientBoostingClassifier())
        ], voting='soft')
               

    def fit(self,X,X_GCN,y):
        # 划分训练集和验证集 --> 阈值计算
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        train_model3 = treat_data_autoglun(X_train,y_train,self.label)
        test_model3 = treat_data_autoglun(X_temp,y_temp,self.label)
        #y_temp.to_excel('y_temp.xlsx', index=True)  # index=False 不保存索引列
        #test_model3.to_excel('test_model3.xlsx', index=True)  # index=False 不保存索引列

        assert train_model3[self.label].notnull().all(), "标签列存在缺失值！"
        assert not np.isinf(train_model3[self.label]).any(), "标签列存在无穷值！"
        
        # 计算最小代价阈值--计算方式2
        model1 = self.model1.fit(X_train, y_train)
        model2 = self.model2.fit(X_train, y_train)
        #model3 = self.model3.fit(train_model3)
        y_pred1 = model1.predict_proba(X_temp)[:,1]
        y_pred2 = model2.predict_proba(X_temp)[:,1]
        #y_pred3 = model3.predict_proba(test_model3)[1]

        thres_model1 = self.compute_threshold(y_pred1, y_temp)
        print('Model1 计算的最佳阈值:', thres_model1) 
        thres_model2 = self.compute_threshold(y_pred2, y_temp)
        print('Model2 计算的最佳阈值:', thres_model2) 
        #thres_model3 = self.compute_threshold(y_pred3, y_temp)
        #print('Model3 计算的最佳阈值:', thres_model3)

        self.thres = (thres_model1 + thres_model2) / 2
        print('最终阈值:', self.thres)

        # 生成元特征
        X_model3 = treat_data_autoglun(X,y,self.label)
        model1_pred = cross_val_predict(self.model1, X, y, cv=5, method='predict_proba')[:,1]
        model2_pred = cross_val_predict(self.model2, X, y, cv=5, method='predict_proba')[:,1]
        #model3_pred = cross_val_predict(self.model3, X_model3, cv=5, method='predict_proba')[1]
        self.model3.fit(
            X_model3,
            num_bag_folds=5,  # 5折交叉验证
            hyperparameters = self.hyperparameters,
            hyperparameter_tune_kwargs= self.hyperparameter_tune_kwargs)  # 先五折交叉验证，并最终输出全量训练模型
        
        # 获取交叉验证预测结果
        model3_pred = self.model3.predict_proba(X_model3)[1]
        self.model4 = self.fit_GCD(X_GCN)
        model4_output = self.model4(X_GCN)
        model4_pred = torch.sigmoid(model4_output)  # 获取概率值

        meta_X = np.column_stack([model1_pred, model2_pred, model3_pred,model4_pred])
        
        # 训练元模型
        self.meta.fit(meta_X, y)
        # 全量训练基模型
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        """
        self.model3.fit(X_model3,
                         hyperparameters = self.hyperparameters,
                        hyperparameter_tune_kwargs= self.hyperparameter_tune_kwargs)
        """ 
        
    def predict(self, X_test, X_test_GCN, y_test):
        test_model3 = treat_data_autoglun(X_test,y_test,self.label)
        test_loader = DataLoader(X_test_GCN, batch_size=32)
        # 基模型输出概率
        model1_output = self.model1.predict_proba(X_test)[:,1]
        model2_output = self.model2.predict_proba(X_test)[:,1]
        model3_output = self.model3.predict_proba(test_model3)[1]
        model4_output = torch.sigmoid(self.model4(test_loader))  # 获取概率值
        ensemble_output = self.meta.predict_proba(np.column_stack([model1_output, model2_output,model3_output,model4_output]))[:,1]
        #print('model1_output:', model1_output)
        #print('model2_output:', model2_output)
        #print('ensemble_output:', ensemble_output)

        # 阈值设置
        """
        fpr, tpr, thresholds = roc_curve(y_test, ensemble_output)
        optimal_idx = np.argmax(tpr - fpr)
        thres = thresholds[optimal_idx]
        """
        """
        thres = self.compute_threshold(ensemble_output, y_test)
        print('计算的最佳阈值:', thres) 
        """
        
        # 将概率转换为标签形式
        thres = self.thres
        #thres = 0.5       
        model1_labels = (model1_output >= thres).astype(int)
        model2_labels = (model2_output >= thres).astype(int)
        model3_labels = (model3_output >= thres).astype(int)
        model4_labels = (model4_output >= thres).astype(int)
        ensemble_labels = (ensemble_output >= thres).astype(int)
        print('基于计算阈值，model1 f1:', f1_score(y_test, model1_labels))
        print('基于计算阈值，model2 f1:', f1_score(y_test, model2_labels))
        print('基于计算阈值，model3 f1:', f1_score(y_test, model3_labels))
        print('基于计算阈值，model4 f1:', f1_score(y_test, model4_labels))
        print('基于计算阈值，ensemble f1:', f1_score(y_test, ensemble_labels))
        """
        print('基于计算阈值，model1 acc:', self.multi_output_accuracy(y_test, model1_labels))
        print('基于计算阈值，model2 acc:', self.multi_output_accuracy(y_test, model2_labels))
        print('基于计算阈值，ensemble acc:', self.multi_output_accuracy(y_test, ensemble_labels))
        """
        
        return ensemble_labels
    
    def fit_GCD(self, train_data):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()
        # 获取实际样本数量
        n_samples = len(data_list)
        print(f"总样本数量: {n_samples}")
        # 修改批次大小，使用较小的批次
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        
        # 使用较小的学习率
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 添加权重衰减
        # 修改模型训练部分，添加学习率调度和早停
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        # 使用较小的学习率
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        # 添加早停机制
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model4.to(device)  # 确保模型在目标设备
        # 训练循环
        losses = []
        for epoch in tqdm(range(500), desc="训练进度"):  # 增加训练轮数
            model.train()
            total_loss = 0
            for data in train_loader:
                data = data.to(device)  # 将整个数据对象（含节点特征、边索引等）转移到设备
                optimizer.zero_grad()
                out = model(data)
                target = data.y.view(-1, 1).float()
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
                torch.save(model.state_dict(), 'F:/03--python/01__MyItems/WIDS/GNNModel/best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'早停于第 {epoch} 轮，最佳损失: {best_loss:.4f}')
                    break
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
        return model

    def multi_output_accuracy(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 处理单任务（一维）和多任务（二维）场景
        if y_true.ndim == 1:
            return accuracy_score(y_true, y_pred)
        else:
            return np.mean([
                accuracy_score(y_true[:, i], y_pred[:, i]) 
                for i in range(y_true.shape[1])
            ])
        

    # 阈值计算迁移至验证集（避免数据泄露）
    def compute_threshold(self, probs, y_true):
        costs = []
        thresholds = np.linspace(0, 1, 100)
        for thresh in thresholds:
            y_pred = (probs >= thresh).astype(int)
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            costs.append(5 * fn + 1 * fp)  # 代价敏感权重
        thres = thresholds[np.argmin(costs)]
        if thres == 0 or thres == 1:
            thres = 0.5
            print(' 阈值计算失败，设置为默认值 0.5) ')  
        return thres
    