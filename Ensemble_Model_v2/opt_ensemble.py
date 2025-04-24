# 2025/04/20


import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from EnsembleModel_v4 import EnsembleModel # type: ignore
from autogluon.common import space

def Opt_Ensemble(X_train, X_train_GCN,y_train, X_test, X_test_GCN,y_test,metadata_dim,label):
    # XGBoost参数空间（保持原样）
    xgb_param_space = {
        'n_estimators': Integer(100, 300),
        'learning_rate': Real(0.001, 0.1, prior='log-uniform'),
        'max_depth': Integer(3, 8),
        'subsample': Real(0.7, 0.9),
        'gamma': Real(1, 10),
        'reg_alpha': Real(1, 50),   # L1正则化系数
        'reg_lambda': Real(1, 50),   # L2正则化系数
        'colsample_bytree': Real(0.6, 0.8),
        'min_child_weight': Integer(5, 20)
    }

    # 新增随机森林参数空间
    rf_param_space = {
        'n_estimators': Integer(400, 800),
        'max_depth': Integer(5, 20),
        'min_samples_split': Real(0.01, 0.5),
        'max_features': Categorical(['sqrt', 'log2'])
    }

    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 增加交叉验证折数
    pos_weight = np.sqrt((y_train == 0).sum() / (y_train == 1).sum())
    print('pos_weight:', pos_weight)

    base_model_xgb = XGBClassifier(
        scale_pos_weight = pos_weight,
        tree_method='hist',
        eval_metric='logloss',
        random_state=42
    )
    xgb_optim = BayesSearchCV(
        estimator=base_model_xgb,
        search_spaces=xgb_param_space,
        n_iter=100,  # 增加迭代次数
        cv=cv,  # 使用交叉验证
        scoring='f1_weighted',
        optimizer_kwargs={'n_initial_points': 20},
        random_state=42,
        n_jobs=-1
    )

    rf_optim = BayesSearchCV(
        RandomForestClassifier(class_weight="balanced"),
        rf_param_space,
        n_iter=100,
        cv = cv,
        scoring='f1_weighted'
    )

    nn_options = {  # specifies non-default hyperparameter values for neural network models
        'num_epochs': 10,  # number of training epochs (controls training time of NN models)
        'learning_rate': space.Real(1e-4, 1e-2, default=5e-4, log=True),  # learning rate used in training (real-valued hyperparameter searched on log-scale)
        'activation': space.Categorical('relu', 'softrelu', 'tanh'),  # activation function used in NN (categorical hyperparameter, default = first entry)
        'dropout_prob': space.Real(0.0, 0.5, default=0.1),  # dropout probability (real-valued hyperparameter)
        'scale_pos_weight': space.Real(1, 3, default=0.2),
    }

    gbm_options = {  # specifies non-default hyperparameter values for lightGBM gradient boosted trees
        'num_boost_round': 100,  # number of boosting rounds (controls training time of GBM models)
        'num_leaves': space.Int(lower=26, upper=66, default=36),  # number of leaves in trees (integer hyperparameter)
    }
    hyperparameters = {  # hyperparameters of each model type
                    'GBM': gbm_options,
                    'XGB': gbm_options,
                    'NN_TORCH': nn_options,  # NOTE: comment this line out if you get errors on Mac OSX
                    }  # When these keys are missing from hyperparameters dict, no models of that type are trained

    #time_limit = 20*60  # train various models for ~2 min
    num_trials = 5  # try at most 5 different hyperparameter configurations for each type of model
    search_strategy = 'auto'  # to tune hyperparameters using random search routine with a local scheduler

    hyperparameter_tune_kwargs = {  # HPO is not performed unless hyperparameter_tune_kwargs is specified
        'num_trials': num_trials,
        'scheduler' : 'local',
        'searcher': search_strategy,
    }  # Refer to TabularPredictor.fit docstring for all valid values

    # 并行优化
    print("开始并行优化 ...")
    xgb_optim.fit(X_train, y_train)
    rf_optim.fit(X_train, y_train)
    print('最佳xgb参数：', xgb_optim.best_params_)
    print('最佳xgb分数：', xgb_optim.best_score_)
    print('最佳rf参数：', rf_optim.best_params_)
    print('最佳rf分数：', rf_optim.best_score_)

    # 构建堆叠模型
    print("构建堆叠模型...")
    stacked_model = EnsembleModel(
        xgb_params=xgb_optim.best_params_,
        rf_params=rf_optim.best_params_,
        hyperparameters=hyperparameters,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        metadata_dim = metadata_dim,
        label = label
    )
    stacked_model.fit(X_train, X_train_GCN,y_train)
    y_pred = stacked_model.predict(X_test, X_test_GCN, y_test)
    
    return y_pred,stacked_model