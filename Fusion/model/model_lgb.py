import os
import pandas as pd
import lightgbm as lgb
import warnings


warnings.filterwarnings("ignore")


def model_lgbm(train, validate, test, return_model=False):
    """
    修改后的LightGBM模型函数

    参数:
    - train: 训练数据
    - validate: 验证数据
    - test: 测试数据
    - return_model: 是否返回模型对象

    返回:
    - 如果 return_model=False: (result, feat_importance, params)
    - 如果 return_model=True: (result, feat_importance, params, model)
    """
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': 2,
        'learning_rate': 0.01,
        'max_depth': 5,
        'min_child_weight': 1,
        'lambda_l1': 1,
        'colsample_bytree': 0.7,
        'subsample': 0.9,
        'bagging_freq': 1
    }

    dtrain = lgb.Dataset(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=train['label'])
    dvalid = lgb.Dataset(validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1),
                         label=validate['label'])

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2700,
        valid_sets=[dtrain, dvalid],
        valid_names=['train', 'validate'],
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=True)]
    )

    predict = model.predict(test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)

    # 特征重要性
    feat_importance = pd.DataFrame({
        'feature_name': model.feature_name(),
        'importance': model.feature_importance(importance_type='split')
    })
    feat_importance.sort_values(['importance'], ascending=False, inplace=True)

    if return_model:
        return result, feat_importance, params, model
    else:
        return result, feat_importance, params


def predict_with_lgbm_model(model, test_data):
    """
    使用已训练的LightGBM模型进行预测
    """
    predict = model.predict(test_data.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test_data[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)
    return result


def log_training_result(params, auc, log_path='lgbm_training_log.csv'):
    """将本次训练的参数与AUC记录到CSV文件"""
    # 构建日志记录行
    record = params.copy()
    record['auc'] = auc
    # 将参数转为DataFrame
    df = pd.DataFrame([record])
    # 如果文件存在则追加,否则新建
    if os.path.exists(log_path):
        df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df.to_csv(log_path, index=False)
    print(f'✅ 已记录训练结果到 {log_path}')


if __name__ == '__main__':
    off_train = pd.read_csv('../dataset/train.csv')

    off_test = pd.read_csv('../dataset/test.csv')
    off_validate = pd.read_csv('../dataset/validate.csv')
    train = off_train.copy()
    test = off_test.copy()
    validate = off_validate.copy()

    '''
    Drop_feature = ['history_user_before_receive_Coupon_cnt', 'history_user_after_receive_all_cnt',
                    'history_user_before_receive_all_cnt',
                    'history_user_Coupon_Date_received_cnt', 'history_user_Date_received_cnt']

    train = train.drop(Drop_feature, axis=1)
    test = test.drop(Drop_feature, axis=1)
    validate = validate.drop(Drop_feature, axis=1)
    '''
    print(train.shape)
    '''
    result_off, feat_importance_off, params = model_lgbm(train, validate, validate.drop(['label'], axis=1))
    auc = coupon_group_auc(validate['label'], result_off['prob'], validate['Coupon_id'])
    print(auc)
    print(feat_importance_off)
    log_training_result(params, auc)
    '''

    big_train = pd.read_csv('big_train_norm2.csv')
    result, feat_importance, params = model_lgbm(big_train, validate, test)
    result.to_csv(r'../model_result/result_lgb.csv', index=False, header=None)
    print(feat_importance)