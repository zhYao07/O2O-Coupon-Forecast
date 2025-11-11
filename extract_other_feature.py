import numpy as np
import pandas as pd
import os

def get_week_feature(label_field):
    # 提取星期几特征

    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)  # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Date_received'] = data['Date_received'].map(int)

    data['week'] = data['date_received'].map(lambda x: x.weekday())  # 星期几
    data['is_weekend'] = data['week'].map(lambda x: 1 if x == 5 or x == 6 else 0)  # 是否周末
    data = pd.concat([data, pd.get_dummies(data['week'], prefix='week')], axis=1)  # one-hot编码

    data.index = range(len(data))

    return data


def get_distance_feature(label_field):
    # 提取距离特征

    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)  # Coupon_id列中存在np.nan会导致float
    data['Date_received'] = data['Date_received'].map(int)

    # 距离是否为空（-1表示无距离信息）
    data['is_distance_null'] = data['Distance'].map(lambda x: 1 if x == -1 else 0)

    # 只对有效距离（0~10）进行独热编码
   # 只对 0~10 的 Distance 做独热编码
    valid_dist = data['Distance'].where(data['Distance'].between(0, 10), np.nan)
    one_hot = pd.get_dummies(valid_dist, prefix='distance')

# 拼回原表
    data = pd.concat([data, one_hot], axis=1)



    data.index = range(len(data))

    return data


def get_other_feature(label_field):
    # 提取其他特征

    data = label_field.copy()

    data = get_week_feature(data)
#    data = get_distance_feature(data)

    return data