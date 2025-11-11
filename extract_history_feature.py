import numpy as np
import pandas as pd
import os

def get_history_user_feature(history_field, label_field):
    data = history_field.copy()
    label_data = label_field.copy()

    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1

    # 1. 用户领取优惠券数
    keys = ['User_id']
    feat = data.groupby(keys)['cnt'].count().reset_index()
    feat.rename(columns={'cnt': 'history_user_received_cnt'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    label_data['history_user_received_cnt'].fillna(0, inplace=True)

    # 2. 用户未核销数
    feat = data[data['label'] == 0].groupby(keys)['cnt'].count().reset_index()
    feat.rename(columns={'cnt': 'history_user_not_consume_cnt'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    label_data['history_user_not_consume_cnt'].fillna(0, inplace=True)

    # 3. 用户核销数
    feat = data[~data['Date'].isnull()].groupby(keys)['cnt'].count().reset_index()
    feat.rename(columns={'cnt': 'history_user_consume_cnt'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    label_data['history_user_consume_cnt'].fillna(0, inplace=True)

    # 4. 用户核销率
    label_data['history_user_consume_rate'] = label_data['history_user_consume_cnt'] / (label_data['history_user_received_cnt'] + 1)

    # 5. 用户领取不同商家数
    feat = data.groupby(keys)['Merchant_id'].nunique().reset_index()
    feat.rename(columns={'Merchant_id': 'history_user_received_differ_Merchant_cnt'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    label_data['history_user_received_differ_Merchant_cnt'].fillna(0, inplace=True)

    # 6. 用户对领券商家15天内的核销数
    keys = ['User_id', 'Merchant_id']
    feat = data[data['label'] == 1].groupby(keys)['cnt'].count().reset_index()
    feat.rename(columns={'cnt': 'history_user_Merchant_consume_in_15days_cnt'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    label_data['history_user_Merchant_consume_in_15days_cnt'].fillna(0, inplace=True)

    # 7. 用户当天领取优惠券数
    keys = ['User_id', 'Date_received']
    feat = data.groupby(keys)['cnt'].count().reset_index()
    feat.rename(columns={'cnt': 'history_user_Date_received_cnt'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')
   # label_data['history_user_Date_received_cnt'].fillna(0, inplace=True)

    # 8. 用户当天领取特定优惠券数
    keys = ['User_id', 'Coupon_id', 'Date_received']
    feat = data.groupby(keys)['cnt'].count().reset_index()
    feat.rename(columns={'cnt': 'history_user_Coupon_Date_received_cnt'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')

 #   label_data['history_user_Coupon_Date_received_cnt'].fillna(0, inplace=True)


    # --------------------------0.8047

    #用户领取优惠券种类数目(低)
    '''
    keys = ['User_id']
    feat = data.groupby(keys)['Coupon_id'].nunique().reset_index()
    feat.rename(columns={'Coupon_id': 'history_field_User_received_differ_Coupon_cnt'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    label_data['history_field_User_received_differ_Coupon_cnt'].fillna(0, inplace=True)
    '''




    #10.19 新增特征
    
    #用户在该商家的核销率

    # 在 get_history_user_feature 中添加
    # 用户在特定商家的历史核销率
    keys = ['User_id', 'Merchant_id']
    temp_received = data.groupby(keys)['cnt'].count().reset_index()
    temp_consumed = data[data['label']==1].groupby(keys)['cnt'].count().reset_index()
    feat = pd.merge(temp_received, temp_consumed, on=keys, how='left', suffixes=('_total', '_consume'))
    feat['history_user_merchant_consume_rate'] = feat['cnt_consume'] / (feat['cnt_total'] + 1)
    feat = feat[['User_id', 'Merchant_id', 'history_user_merchant_consume_rate']]
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    #-----------------------------------------------------------------会掉分
    label_data['history_user_merchant_consume_rate'].fillna(0, inplace=True)

    # 用户对不同折扣率的偏好（历史核销率）
    keys = ['User_id', 'discount_rate']
    temp_received = data.groupby(keys)['cnt'].count().reset_index()
    temp_consumed = data[data['label']==1].groupby(keys)['cnt'].count().reset_index()
    feat = pd.merge(temp_received, temp_consumed, on=keys, how='left', suffixes=('_total', '_consume'))
    feat['history_user_discount_consume_rate'] = feat['cnt_consume'] / (feat['cnt_total'] + 1)
    feat = feat[['User_id', 'discount_rate', 'history_user_discount_consume_rate']]
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    #-----------------------------------------------------------------3100 0.8033 3150 0.8029
    label_data['history_user_discount_consume_rate'].fillna(0, inplace=True)

    # 用户平均核销距离
    keys = ['User_id']
    feat = data[(data['label']==1) & (data['Distance'].notna())].groupby(keys)['Distance'].mean().reset_index()
    feat.rename(columns={'Distance': 'history_user_avg_consume_distance'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    #------------------------------------------------------------  3150 0.8019
    label_data['history_user_avg_consume_distance'].fillna(-1, inplace=True)


    #------------
    # 9.用户平均核销每个商家多少张优惠券
    '''
    temp = data[data['label'] == 1].groupby(['User_id', 'Merchant_id'])['cnt'].count().reset_index()
    feat = temp.groupby(['User_id'])['cnt'].mean().reset_index()
    feat.rename(columns={'cnt': 'history_user_avg_consume_Coupon_per_Merchant'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=['User_id'], how='left')
    label_data['history_user_avg_consume_Coupon_per_Merchant'].fillna(0 , inplace=True)
'''



    # 0.7980 后新增特征
    # 用户-优惠券的历史核销率（提）
    keys = ['User_id', 'Coupon_id']
    temp_received = data.groupby(keys)['cnt'].count().reset_index()
    temp_consumed = data[data['label'] == 1].groupby(keys)['cnt'].count().reset_index()
    feat = pd.merge(temp_received, temp_consumed, on=keys, how='left', suffixes=('_total', '_consume'))
    feat['history_user_coupon_consume_rate'] = feat['cnt_consume'] / (feat['cnt_total'] + 1)
    feat = feat[['User_id', 'Coupon_id', 'history_user_coupon_consume_rate']]
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    #----------------------------------------------------------3100 0.8017
    label_data['history_user_coupon_consume_rate'].fillna(0, inplace=True)

    # 用户核销时间偏好（平均核销间隔天数）（提）
    keys = ['User_id']
    feat = data[data['label'] == 1].groupby(keys).apply(
        lambda x: (x['date'] - x['date_received']).dt.days.mean()
    ).reset_index()
    feat.rename(columns={0: 'history_user_avg_consume_lag'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    #--------------------------------------------------------------- 3100 0.8034
    label_data['history_user_avg_consume_lag'].fillna(-1, inplace=True)


    #0.8044后新增
    # 用户对特定商家特定优惠券的偏好
    '''
    keys = ['User_id', 'Merchant_id', 'Coupon_id']
    temp_received = data.groupby(keys)['cnt'].count().reset_index()
    temp_consumed = data[data['label'] == 1].groupby(keys)['cnt'].count().reset_index()
    feat = pd.merge(temp_received, temp_consumed, on=keys, how='left', suffixes=('_total', '_consume'))
    feat['history_user_merchant_coupon_consume_rate'] = feat['cnt_consume'] / (feat['cnt_total'] + 1)
    label_data = pd.merge(label_data,
                          feat[['User_id', 'Merchant_id', 'Coupon_id', 'history_user_merchant_coupon_consume_rate']],
                          on=keys, how='left').fillna(0)
    '''
    #用户对特定商家特定优惠券的领券数（掉）
    '''
    keys = ['User_id', 'Merchant_id', 'Coupon_id']
    temp_received = data.groupby(keys)['cnt'].count().reset_index()
    temp_received.rename(columns={'cnt': 'history_field_User_Merchant_Couppon_received_cnt'}, inplace=True)

    label_data = pd.merge(label_data, temp_received, on=keys, how='left')
    label_data.fillna(0, inplace=True)
    '''
    #----------------------------------------------
    # 用户历史核销距离的变异系数



    data = data.sort_values(['User_id', 'Date_received'])
    # 9-12 与领取顺序相关
    temp = data[['User_id', 'Coupon_id', 'Date_received']].drop_duplicates()
    temp['history_user_before_receive_all_cnt'] = temp.groupby('User_id').cumcount()
    temp['history_user_after_receive_all_cnt'] = temp.groupby('User_id')['Date_received'].transform('count') - 1 - temp['history_user_before_receive_all_cnt']
    temp['history_user_before_receive_Coupon_cnt'] = temp.groupby(['User_id', 'Coupon_id']).cumcount()
    temp['history_user_after_receive_Coupon_cnt'] = temp.groupby(['User_id', 'Coupon_id'])['Coupon_id'].transform('count') - 1 - temp['history_user_before_receive_Coupon_cnt']

    label_data = pd.merge(label_data, temp, on=['User_id', 'Coupon_id', 'Date_received'], how='left')
 #   label_data.fillna(0, inplace=True)



    return label_data


def get_history_merchant_feature(history_field, label_field):
    data = history_field.copy()
    label_data = label_field.copy()

    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1

    # 1. 商家被领取优惠券数目
    keys = ['Merchant_id']
    feat = data.groupby(keys)['cnt'].count().reset_index()
    feat.rename(columns={'cnt': 'history_merchant_received_cnt'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    label_data['history_merchant_received_cnt'].fillna(0, inplace=True)

    # 2. 商家被不同客户领取次数
    feat = data.groupby(keys)['User_id'].nunique().reset_index()
    feat.rename(columns={'User_id': 'history_merchant_differ_User_received_cnt'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    label_data['history_merchant_differ_User_received_cnt'].fillna(0, inplace=True)

    # 3. 商家的券被核销次数
    feat = data[data['label'] == 1].groupby(keys)['cnt'].count().reset_index()
    feat.rename(columns={'cnt': 'history_merchant_consume_cnt'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    label_data['history_merchant_consume_cnt'].fillna(0, inplace=True)

    # 4. 商家券核销率
    label_data['history_merchant_consume_rate'] = label_data['history_merchant_consume_cnt'] / (label_data['history_merchant_received_cnt'] + 1)

    # 5. 商家券未核销次数
    feat = data[data['label'] == 0].groupby(keys)['cnt'].count().reset_index()
    feat.rename(columns={'cnt': 'history_merchant_not_consume_cnt'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    label_data['history_merchant_not_consume_cnt'].fillna(0, inplace=True)

    # 6. 商家提供不同优惠券数
    feat = data.groupby(keys)['Coupon_id'].nunique().reset_index()
    feat.rename(columns={'Coupon_id': 'history_merchant_differ_Coupon_cnt'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    label_data['history_merchant_differ_Coupon_cnt'].fillna(0, inplace=True)


    # 0.7980 后新增特征
    #商家优惠券历史核销率
    keys = ['Merchant_id', 'Coupon_id']
    temp_received = data.groupby(keys)['cnt'].count().reset_index()
    temp_consumed = data[data['label'] == 1].groupby(keys)['cnt'].count().reset_index()
    feat = pd.merge(temp_received, temp_consumed, on=keys, how='left', suffixes=('_total', '_consume'))
    feat['history_merchant_coupon_consume_rate'] = feat['cnt_consume'] / (feat['cnt_total'] + 1)
    feat = feat[['Merchant_id', 'Coupon_id', 'history_merchant_coupon_consume_rate']]
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    label_data['history_merchant_coupon_consume_rate'].fillna(0, inplace=True)

    '''
    #7.商家优惠券被核销的平均折扣率
    feat = data[data['label'] == 1].groupby(keys)['discount_rate'].mean().reset_index()
    feat.rename(columns={'discount_rate': 'history_merchant_avg_consume_discount_rate'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    label_data['history_merchant_avg_consume_discount_rate'].fillna(0, inplace=True)
    '''
    '''
    #8.商家优惠券平均每个用户核销多少张
    temp = data[data['label'] == 1].groupby(['Merchant_id', 'User_id'])['cnt'].count().reset_index()
    feat = temp.groupby(['Merchant_id'])['cnt'].mean().reset_index()
    feat.rename(columns={'cnt': 'history_merchant_avg_consume_Coupon_per_User'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=['Merchant_id'], how='left')
    label_data['history_merchant_avg_consume_Coupon_per_User'].fillna(0 , inplace=True)
'''
    '''
    # 9. 商家被核销过的不同优惠券数量
    feat = data[data['label'] == 1].groupby(keys)['Coupon_id'].nunique().reset_index()
    feat.rename(columns={'Coupon_id': 'history_merchant_differ_Coupon_consume_cnt'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    label_data['history_merchant_differ_Coupon_consume_cnt'].fillna(0, inplace=True)


    # 10 商家平均每种优惠券核销多少张
    temp = data[data['label'] == 1].groupby(['Merchant_id', 'Coupon_id'])['cnt'].count().reset_index()
    feat = temp.groupby(['Merchant_id'])['cnt'].mean().reset_index()
    feat.rename(columns={'cnt': 'history_merchant_avg_consume_Coupon_per_Coupon'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=['Merchant_id'], how='left')
    label_data['history_merchant_avg_consume_Coupon_per_Coupon'].fillna(0 , inplace=True)
    '''
    '''
    #11. 商家被核销优惠券中的平均距离
    feat = data[data['label'] == 1].groupby(['Merchant_id'])['Distance'].mean().reset_index()
    feat.rename(columns={'Distance': 'history_merchant_avg_consume_distance'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=['Merchant_id'], how='left')
    label_data['history_merchant_avg_consume_distance'].fillna(-1 , inplace=True)
    '''


    return label_data


def get_history_field_coupon_feature(history_field, label_field):
    data = history_field.copy()
    label_data = label_field.copy()

    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1

    # 1. 优惠券被领取次数
    keys = ['Coupon_id']
    feat = data.groupby(keys)['cnt'].count().reset_index()
    feat.rename(columns={'cnt': 'history_coupon_received_cnt'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    label_data['history_coupon_received_cnt'].fillna(0, inplace=True)

    # 2. 优惠券15天内被核销次数
    feat = data[data['label'] == 1].groupby(keys)['cnt'].count().reset_index()
    feat.rename(columns={'cnt': 'history_coupon_received_and_consume_cnt_15'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    label_data['history_coupon_received_and_consume_cnt_15'].fillna(0, inplace=True)

    # 3. 核销率
    label_data['history_coupon_received_and_consume_rate_15'] = label_data['history_coupon_received_and_consume_cnt_15'] / (label_data['history_coupon_received_cnt'] + 1)

    # 4. 核销平均时间间隔
    feat = data[data['label'] == 1].groupby(keys).apply(lambda x: (x['date'] - x['date_received']).dt.days.mean()).reset_index()
    feat.rename(columns={0: 'history_coupon_avg_consume_gap_15'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    label_data['history_coupon_avg_consume_gap_15'].fillna(-1, inplace=True)

    # 5. 满减券最低消费中位数
    pivot = pd.pivot_table(data[data['is_manjian'] == 1], index=keys, values='min_cost_of_manjian', aggfunc=np.median)
    pivot = pd.DataFrame(pivot).rename(columns={'min_cost_of_manjian': 'history_coupon_median_of_min_cost_of_manjian'}).reset_index()
    label_data = pd.merge(label_data, pivot, on=keys, how='left')
    label_data['history_coupon_median_of_min_cost_of_manjian'].fillna(0, inplace=True)

    return label_data


def get_history_field_user_coupon_feature(history_field, label_field):
    data = history_field.copy()
    label_data = label_field.copy()

    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1

    # 1. 用户领取特定优惠券数
    keys = ['User_id', 'Coupon_id']
    feat = data.groupby(keys)['cnt'].count().reset_index()
    feat.rename(columns={'cnt': 'history_user_coupon_received_cnt'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    label_data['history_user_coupon_received_cnt'].fillna(0, inplace=True)

    # 2. 用户核销特定优惠券数
    feat = data[data['label'] == 1].groupby(keys)['cnt'].count().reset_index()
    feat.rename(columns={'cnt': 'history_user_coupon_consume_cnt'}, inplace=True)
    label_data = pd.merge(label_data, feat, on=keys, how='left')
    label_data['history_user_coupon_consume_cnt'].fillna(0, inplace=True)

#--------------------------------
    #0.8044后新增特征

    '''
    # ✅ 混合特征：当前折扣率与用户历史偏好折扣率的差异(掉)
    keys = ['User_id']
    feat = data[data['label'] == 1].groupby(keys)['discount_rate'].mean().reset_index()
    feat.rename(columns={'discount_rate': 'history_user_avg_consume_discount'}, inplace=True)
    #  label_feat = pd.merge(label_feat, feat, on=keys, how='left').fillna(-1)

    label_data['discount_diff_from_history'] = (
            label_data['discount_rate'] - feat['history_user_avg_consume_discount']
    )
    '''

    return label_data




def get_history_field_feature(history_field, label_field):
    data = label_field.copy()
    data = get_history_user_feature(history_field, data)
    data = get_history_merchant_feature(history_field, data)
    data = get_history_field_coupon_feature(history_field, data)
    data = get_history_field_user_coupon_feature(history_field, data)



    return data
