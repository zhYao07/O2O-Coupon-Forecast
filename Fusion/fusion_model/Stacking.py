import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

from model.model_lgb import model_lgbm, predict_with_lgbm_model
from model.model_xgb import model_xgb, predict_with_model
from model.xgb_2500 import model_xgb2500, predict_with_model2
from model.xgb_4000 import model_xgb4000, predict_with_model4

def coupon_group_auc(y_true,y_pred,coupon_id):

    '''
    计算ROC曲线下围成的面积
    横坐标：FPR(假正率) FPR = FP / TN + FP
    纵坐标：TPR(真正率) TPR = TP / TP + FN

    :param y_true:
    :param y_pred:
    :param coupon_id:
    :return:
    '''

    df = pd.DataFrame({'coupon_id':coupon_id,'y_true':y_true,'y_pred':y_pred})

    #每一组coupon的auc值
    auc_list = []

    for coupon_id, group in df.groupby('coupon_id'):  #group为子表格，包含着当前coupon_id的所有行
        # 如果某个优惠券只有一个类别，roc_auc_score会报错，需要跳过
        if len(group['y_true'].unique()) < 2:
            continue

        auc = roc_auc_score(group['y_true'], group['y_pred'])
        auc_list.append(auc)

    mean_auc = np.mean(auc_list)

    return mean_auc
# 加载数据
off_train = pd.read_csv(r'C:\Users\86188\Desktop\Fusion\dataset\train.csv')
off_test = pd.read_csv(r'C:\Users\86188\Desktop\Fusion\dataset\test.csv')
off_validate = pd.read_csv(r'C:\Users\86188\Desktop\Fusion\dataset\validate.csv')
train = off_train.copy()
validate = off_validate.copy()
test = off_test.copy()

big_train = pd.concat([train, validate], axis=0)

# Step 1: 用KFold生成OOF预测（meta_train）和测试集预测
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

# 初始化OOF prob数组（用于第二层训练数据）
oof_xgb = np.zeros(len(big_train))
oof_rf = np.zeros(len(big_train))
oof_lgbm = np.zeros(len(big_train))

# 初始化测试集预测数组（用于第二层测试数据）
test_pred_xgb = np.zeros((n_splits, len(test)))
test_pred_rf = np.zeros((n_splits, len(test)))
test_pred_lgbm = np.zeros((n_splits, len(test)))

# 为每个基模型做CV
for fold, (train_idx, valid_idx) in enumerate(kf.split(big_train)):
    fold_train = big_train.iloc[train_idx].reset_index(drop=True)
    fold_valid = big_train.iloc[valid_idx].reset_index(drop=True)
    fold_valid_no_label = fold_valid.drop(['label'], axis=1).reset_index(drop=True)

    print(f"Fold {fold + 1} started...")

    # XGBoost - 训练一次，预测验证集和测试集
    result_xgb, _, _, xgb_model = model_xgb(fold_train, fold_valid, fold_valid_no_label, return_model=True)
    oof_xgb[valid_idx] = result_xgb['prob']
    test_result_xgb = predict_with_model(xgb_model, test)
    test_pred_xgb[fold] = test_result_xgb['prob']
    print(f"  XGBoost completed")

    # RandomForest - 训练一次，预测验证集和测试集
    result_rf, _, _, rf_model = model_xgb2500(fold_train, fold_valid, fold_valid_no_label, return_model=True)
    oof_rf[valid_idx] = result_rf['prob']
    test_result_rf = predict_with_model2(rf_model, test)
    test_pred_rf[fold] = test_result_rf['prob']
    print(f"  RandomForest completed")

    # LightGBM - 训练一次，预测验证集和测试集
    result_lgbm, _, _, lgbm_model = model_xgb4000(fold_train, fold_valid, fold_valid_no_label, return_model=True)
    oof_lgbm[valid_idx] = result_lgbm['prob']
    test_result_lgbm = predict_with_model4(lgbm_model, test)
    test_pred_lgbm[fold] = test_result_lgbm['prob']
    print(f"  LightGBM completed")

    print(f"Fold {fold + 1} completed")

# 组装meta_train（第二层训练数据）
meta_train = pd.DataFrame({
    'prob_xgb': oof_xgb,
    'prob_rf': oof_rf,
    'prob_lgbm': oof_lgbm,
    'label': big_train['label'],
    'Coupon_id': big_train['Coupon_id']
})

# 对测试集预测取平均（第二层测试数据）
meta_test = pd.DataFrame({
    'prob_xgb': test_pred_xgb.mean(axis=0),  # 对5折预测取平均
    'prob_rf': test_pred_rf.mean(axis=0),
    'prob_lgbm': test_pred_lgbm.mean(axis=0)
})

print(f"Meta train shape: {meta_train.shape}")
print(f"Meta test shape: {meta_test.shape}")

# 可选: 计算OOF AUC评估
oof_pred = (oof_xgb + oof_rf + oof_lgbm) / 3
cv_auc = coupon_group_auc(meta_train['label'], oof_pred, meta_train['Coupon_id'])
print(f"CV OOF AUC (simple avg): {cv_auc}")

# Step 2: 训练元模型
X_meta_train = meta_train[['prob_xgb', 'prob_rf', 'prob_lgbm']]
y_meta_train = meta_train['label']

data_train = meta_train[['prob_xgb', 'prob_rf', 'prob_lgbm','label']]

data_train.to_csv('stacking_train.csv')

meta_model = LogisticRegression()

meta_model.fit(X_meta_train, y_meta_train)

# Step 3: 预测test
X_meta_test = meta_test[['prob_xgb', 'prob_rf', 'prob_lgbm']]
X_meta_test.to_csv('stacking_test.csv')
pred_test = meta_model.predict_proba(X_meta_test)[:, 1]

# 保存结果
result_stacking = test[['User_id', 'Coupon_id', 'Date_received']].copy()
result_stacking['prob'] = pred_test
result_stacking.to_csv('stacking_result_kfold.csv', index=False, header=None)

print("K-fold Stacking融合完成，结果保存为 stacking_result_kfold.csv")