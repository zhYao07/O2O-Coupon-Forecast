import os
import pandas as pd
import xgboost as xgb
import warnings
import time
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def model_xgb(train,validate,test):

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
    dtrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=train['label'])
    dvalid = xgb.DMatrix(validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1),
                         label=validate['label'])
    dtest  = xgb.DMatrix(test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))

    watchlist = [(dtrain, 'train')]

    model = xgb.train(params, dtrain, num_boost_round=3110, evals=watchlist)

    predict = model.predict(dtest)

    #处理结果
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)

    #特征重要性
    score = model.get_score(importance_type='gain')
    feat_importance = pd.DataFrame({
        'feature_name': list(score.keys()),
        'importance': list(score.values())
    })
    feat_importance.sort_values(['importance'], ascending=False, inplace=True)

    return result, feat_importance,model

def log_training_result(params, auc, log_path='xgb_training_log.csv'):
    """将本次训练的参数与AUC记录到CSV文件"""
    # 构建日志记录行
    record = params.copy()
    record['auc'] = auc

    # 将参数转为DataFrame
    df = pd.DataFrame([record])

    # 如果文件存在则追加，否则新建
    if os.path.exists(log_path):
        df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df.to_csv(log_path, index=False)

    print(f'✅ 已记录训练结果到 {log_path}')

if __name__ == '__main__':
    off_train = pd.read_csv('dataset/train.csv')


    off_test = pd.read_csv('dataset/test.csv')
    off_validate = pd.read_csv('dataset/validate.csv')

    train = off_train.copy()
    test = off_test.copy()
    validate = off_validate.copy()

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
  #  print(big_train.isnull().any())
    print("train_shape:",train.shape)
    #计算训练集正负样本比例：
    print("正样本比例: {:.4f}%".format(big_train['label'].sum() / len(big_train) * 100))
    big_train.to_csv(r'./big_train.csv', index=False)
    start_time = time.time()
    result, feat_importance,model = model_xgb(big_train, validate,test)
    end_time = time.time()
    print(f"Time taken for model_xgb: {end_time - start_time:.2f} seconds")
    result.to_csv(r'./result.csv', index=False, header=None)
    print(feat_importance)

    #保存特征重要性图

    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, importance_type='gain', max_num_features=20, ax=ax)
    ax.set_title('Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig('feature_importance_gain.png')
    plt.close()