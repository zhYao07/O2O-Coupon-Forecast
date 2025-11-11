import pandas as pd
import numpy as np

# 读取三个模型的预测结果CSV
df_xgb = pd.read_csv(r'C:\Users\86188\Desktop\Fusion\model_result\result_xgb.csv', header=None, names=['User_id', 'Coupon_id', 'Date_received', 'prob_xgb'])
df_xgb2500 = pd.read_csv(r'C:\Users\86188\Desktop\Fusion\model_result\result_xgb_2500.csv', header=None, names=['User_id', 'Coupon_id', 'Date_received', 'prob_xgb2500'])
df_xgb4000 = pd.read_csv(r'C:\Users\86188\Desktop\Fusion\model_result\result_xgb_4000.csv', header=None, names=['User_id', 'Coupon_id', 'Date_received', 'prob_xgb4000'])

# 合并DataFrame，便于计算
df_merged = pd.concat([df_xgb, df_xgb2500['prob_xgb2500'], df_xgb4000['prob_xgb4000']], axis=1)

# 计算每个模型的rank（高概率对应低rank值，即rank=1是最高）
df_merged['rank_xgb'] = df_merged['prob_xgb'].rank(ascending=False)
df_merged['rank_xgb2500'] = df_merged['prob_xgb2500'].rank(ascending=False)

# 计算平均rank
df_merged['avg_rank'] = (0.4 * df_merged['rank_xgb'] + 0.3 * df_merged['rank_xgb2500'] + 0.3 * df_merged['rank_xgb4000'])
df_merged['prob'] = 1 - (df_merged['avg_rank'] - 1) / (len(df_merged) - 1)

# 只保留需要的列
df_fused = df_merged[['User_id', 'Coupon_id', 'Date_received', 'prob']]

# 保存融合结果（无索引、无表头）
df_fused.to_csv('fused_rank_result.csv', index=False, header=None)

print("基于Rank的融合完成，结果保存为 fused_rank_result.csv")