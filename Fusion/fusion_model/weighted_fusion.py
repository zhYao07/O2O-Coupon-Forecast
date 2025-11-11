import os
import numpy as np
import pandas as pd


# 读取三个模型的预测结果CSV
df_xgb = pd.read_csv(r'C:\Users\86188\Desktop\Fusion\model_result\result_xgb.csv', header=None, names=['User_id', 'Coupon_id', 'Date_received', 'prob_xgb'])
df_xgb_2500 = pd.read_csv(r'C:\Users\86188\Desktop\Fusion\model_result\result_xgb_2500.csv', header=None, names=['User_id', 'Coupon_id', 'Date_received', 'prob_xgb2500'])
df_xgb_4000 = pd.read_csv(r'C:\Users\86188\Desktop\Fusion\model_result\result_xgb_4000.csv', header=None, names=['User_id', 'Coupon_id', 'Date_received', 'prob_xgb4000'])
# 检查数据行数是否一致


# 计算加权融合概率
df_fused = df_xgb.copy()  # 以XGBoost的DataFrame为基础


'''
df_fused['prob'] = (w_xgb * df_xgb['prob_xgb'] +
                    w_xgb2500 * df_xgb_2000['prob_xgb2000'] +
                    w_xgb4000 * df_xgb_4000['prob_xgb4000'])
'''


df_fused['prob'] = (0.5 * df_xgb_2500['prob_xgb2500'] + 0.5 * df_xgb_4000['prob_xgb4000'] )

# 只保留需要的列
df_fused = df_fused[['User_id', 'Coupon_id', 'Date_received', 'prob']]

# 保存融合结果（无索引、无表头）
df_fused.to_csv('weighted_fused_result.csv', index=False, header=None)

print("融合完成，结果保存为 weighted_fused_result.csv")