import os
import pandas as pd
import xgboost as xgb
import warnings


warnings.filterwarnings("ignore")


def model_xgb2500(train, validate, test, return_model=False):
    """
    修改后的XGBoost模型函数

    参数:
    - train: 训练数据
    - validate: 验证数据
    - test: 测试数据
    - return_model: 是否返回模型对象

    返回:
    - 如果 return_model=False: (result, feat_importance, params)
    - 如果 return_model=True: (result, feat_importance, params, model)
    """
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'silent': 1,
              'eta': 0.01,
              'max_depth': 5,
              'min_child_weight': 1,
              'gamma': 0,
              'lambda': 1,
              'colsample_bylevel': 0.7,
              'colsample_bytree': 0.7,
              'subsample': 0.9,
              'scale_pos_weight': 1}

    # 准备训练数据
    dtrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1),
                         label=train['label'])

    # 准备验证数据
    dvalid = xgb.DMatrix(validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1),
                         label=validate['label'])

    # 准备测试数据
    dtest = xgb.DMatrix(test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))

    watchlist = [(dtrain, 'train'), (dvalid, 'validate')]
    model = xgb.train(params, dtrain, num_boost_round=3105, evals=watchlist)

    # 预测测试集
    predict = model.predict(dtest)
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)

    # 特征重要性
    feat_importance = pd.DataFrame(columns=['feature_name', 'importance'])
    feat_importance['feature_name'] = list(model.get_score().keys())
    feat_importance['importance'] = list(model.get_score().values())
    feat_importance.sort_values(['importance'], ascending=False, inplace=True)

    if return_model:
        return result, feat_importance, params, model
    else:
        return result, feat_importance, params


def predict_with_model2(model, test_data):
    """
    使用已训练的模型进行预测
    """
    dtest = xgb.DMatrix(test_data.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))
    predict = model.predict(dtest)
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test_data[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)
    return result