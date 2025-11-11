import os
import pandas as pd
import xgboost as xgb
import warnings
import time

warnings.filterwarnings("ignore")


def model_xgb(train, validate, test, return_model=False):
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
              'colsample_bylevel': 0.6,
              'colsample_bytree': 0.6,
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
    model = xgb.train(params, dtrain, num_boost_round=3110, evals=watchlist)

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


def predict_with_model(model, test_data):
    """
    使用已训练的模型进行预测
    """
    dtest = xgb.DMatrix(test_data.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))
    predict = model.predict(dtest)
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test_data[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)
    return result

if __name__ == '__main__':

    off_train = pd.read_csv('../dataset/train.csv')


    off_test = pd.read_csv('../dataset/test.csv')
    off_validate = pd.read_csv('../dataset/validate.csv')

    train = off_train.copy()
    test = off_test.copy()
    validate = off_validate.copy()

    '''
    drop_features = ['Merchant_Coupon_popularity_rank','history_user_avg_consume_lag']
    train = train.drop(drop_features, axis=1)
    validate = validate.drop(drop_features, axis=1)
    test = test.drop(drop_features, axis=1)
    '''
    print(train.shape)
    print(train.columns)
    '''
    result_off, feat_importance_off,parmas = model_xgb(train,validate, validate.drop(['label'], axis=1))
    auc = coupon_group_auc(validate['label'], result_off['prob'], validate['Coupon_id'])
    print(auc)
    print(feat_importance_off)
    log_training_result(parmas, auc)
    '''




    big_train = pd.concat([train, validate], axis=0)
   # big_train.to_csv(r'./big_train.csv', index=False)
    start_time = time.time()
    result, feat_importance,parma = model_xgb(big_train, validate,test)
    end_time = time.time()
    print(f"Time taken for model_xgb: {end_time - start_time:.2f} seconds")
    result.to_csv(r'../model_result/result_xgb_006.csv', index=False, header=None)
    print(feat_importance)
