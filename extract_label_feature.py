import pandas as pd
import numpy as np

def _counts_in_lookback(df, group_col, date_col, windows):
    res = {w: np.zeros(len(df), dtype=np.int32) for w in windows}
    groups = df.groupby(group_col)
    for name, g in groups:
        idx = g.index.values
        dates = g[date_col].values.astype('datetime64[D]').astype('int')
        for w in windows:
            left = dates - w
            left_idx = np.searchsorted(dates, left, side='left')
            counts = np.arange(len(dates)) - left_idx
            counts[counts < 0] = 0
            res[w][idx] = counts
    return res
def get_label_user_feature(label_field):
    """
    用户领取优惠券特征（10个）
    """
    data = label_field.copy()
    data['cnt'] = 1

    # 1. 用户领取所有优惠券数目
    keys = ['User_id']
    feat = data.groupby(keys)['cnt'].count().reset_index()
    feat.rename(columns={'cnt': 'label_field_User_received_cnt'}, inplace=True)
    data = pd.merge(data, feat, on=keys, how='left')
    data['label_field_User_received_cnt'].fillna(0, inplace=True)

    # 2. 用户领取特定优惠券数目
    keys = ['User_id', 'Coupon_id']
    feat = data.groupby(keys)['cnt'].count().reset_index()
    feat.rename(columns={'cnt': 'label_field_User_Coupon_received_cnt'}, inplace=True)
    data = pd.merge(data, feat, on=keys, how='left')
    data['label_field_User_Coupon_received_cnt'].fillna(0, inplace=True)

    # 3. 用户领取特定商家的优惠券数目
    keys = ['User_id', 'Merchant_id']
    feat = data.groupby(keys)['cnt'].count().reset_index()
    feat.rename(columns={'cnt': 'label_field_User_Merchant_received_cnt'}, inplace=True)
    data = pd.merge(data, feat, on=keys, how='left')
    data['label_field_User_Merchant_received_cnt'].fillna(0, inplace=True)

    # 4. 用户当天领取优惠券数目
    keys = ['User_id', 'Date_received']
    feat = data.groupby(keys)['cnt'].count().reset_index()
    feat.rename(columns={'cnt': 'label_field_User_Date_received_cnt'}, inplace=True)
    data = pd.merge(data, feat, on=keys, how='left')
    data['label_field_User_Date_received_cnt'].fillna(0, inplace=True)

    # 5. 用户当天领取特定优惠券数目
    keys = ['User_id', 'Coupon_id', 'Date_received']
    feat = data.groupby(keys)['cnt'].count().reset_index()
    feat.rename(columns={'cnt': 'label_field_User_Coupon_Date_received_cnt'}, inplace=True)
    data = pd.merge(data, feat, on=keys, how='left')
    data['label_field_User_Coupon_Date_received_cnt'].fillna(0, inplace=True)

    # 6. 用户领取所有优惠券种类数目
    keys = ['User_id']
    feat = data.groupby(keys)['Coupon_id'].nunique().reset_index()
    feat.rename(columns={'Coupon_id': 'label_field_User_received_differ_Coupon_cnt'}, inplace=True)
    data = pd.merge(data, feat, on=keys, how='left')
    data['label_field_User_received_differ_Coupon_cnt'].fillna(0, inplace=True)

    # 按领取时间排序
    data = data.sort_values(['User_id', 'Date_received'])

    # 7. 领取券后又领了多少券
    data['label_field_User_after_receive_all_cnt'] = (
        data.groupby('User_id')['Date_received'].transform('count')- 1 - data.groupby('User_id').cumcount()
    )
    data['label_field_User_after_receive_all_cnt'].fillna(0, inplace=True)

    # 8. 领券前又领了多少券
    data['label_field_User_before_receive_all_cnt'] = data.groupby('User_id').cumcount()
    data['label_field_User_before_receive_all_cnt'].fillna(0, inplace=True)

    # 9. 领券前领了多少特定优惠券
    data['label_field_User_before_receive_Coupon_cnt'] = data.groupby(['User_id', 'Coupon_id']).cumcount()
    data['label_field_User_before_receive_Coupon_cnt'].fillna(0, inplace=True)

    # 10. 领取该券后, 用户领取了多少次这种优惠券
    data['label_field_User_after_receive_Coupon_cnt'] = (
        data.groupby(['User_id', 'Coupon_id'])['Coupon_id'].transform('count')
        - 1
        - data.groupby(['User_id', 'Coupon_id']).cumcount()
    )
    data['label_field_User_after_receive_Coupon_cnt'].fillna(0, inplace=True)

    #新增特征
    # 11. 用户最近一次领取该券距今多少天
    data['days_since_last_receive'] = (data['date_received'] - data.groupby('User_id')['date_received'].shift(1)).dt.days
 #   data['days_since_last_receive'].fillna(0, inplace=True)

    #----------------0.8044后新增特征
    # 用户距离上次领该商家券的天数  (加了掉分)
    '''
    data['days_since_last_Merchant_receive'] = (data['date_received'] - data.groupby(['User_id', 'Merchant_id'])['date_received'].shift(1)).dt.days
    data['days_since_last_Merchant_receive'].fillna(0, inplace=True)
    '''

    # 用户对特定商家特定优惠券的领券数（掉）
    '''
    keys = ['User_id', 'Merchant_id', 'Coupon_id']
    temp_received = data.groupby(keys)['cnt'].count().reset_index()
    temp_received.rename(columns={'cnt': 'label_field_User_Merchant_Couppon_received_cnt'}, inplace=True)

    data = pd.merge(data, temp_received, on=keys, how='left')
    data.fillna(0, inplace=True)
    '''

    #--------------------0.8044后新增特征（从历史特征里提取的）
    #优惠券被领取次数
    '''
    3100 0.8037
    keys = ['Coupon_id']
    feat = data.groupby(keys)['cnt'].count().reset_index()
    feat.rename(columns={'cnt': 'label_field_Coupon_received_cnt'}, inplace=True)
    data = pd.merge(data, feat, on=keys, how='left')
    data['label_field_Coupon_received_cnt'].fillna(0, inplace=True)
    '''

    '''
    #满减券最低消费中位数
    pivot = pd.pivot_table(data[data['is_manjian'] == 1], index=keys, values='min_cost_of_manjian', aggfunc=np.median)
    pivot = pd.DataFrame(pivot).rename(columns={'min_cost_of_manjian': 'label_coupon_median_of_min_cost_of_manjian'}).reset_index()
    data = pd.merge(data, pivot, on=keys, how='left')
    data['label_coupon_median_of_min_cost_of_manjian'].fillna(0, inplace=True)
    '''



    data.drop(['cnt'], axis=1, inplace=True)
    return data


def get_label_Merchant_feature(label_field):
    """
    商家被领取优惠券特征（4个）
    """
    data = label_field.copy()
    data['cnt'] = 1

    # 1. 商家被领取的优惠券数目
    keys = ['Merchant_id']
    feat = data.groupby(keys)['cnt'].count().reset_index()
    feat.rename(columns={'cnt': 'label_field_Merchant_received_cnt'}, inplace=True)
    data = pd.merge(data, feat, on=keys, how='left')

    # 2. 商家被领取的特定优惠券数目
    keys = ['Merchant_id', 'Coupon_id']
    feat = data.groupby(keys)['cnt'].count().reset_index()
    feat.rename(columns={'cnt': 'label_field_Merchant_Coupon_received_cnt'}, inplace=True)
    data = pd.merge(data, feat, on=keys, how='left')

    # 3. 商家被多少不同用户领取
    keys = ['Merchant_id']
    feat = data.groupby(keys)['User_id'].nunique().reset_index()
    feat.rename(columns={'User_id': 'label_field_Merchant_differ_User_received_cnt'}, inplace=True)
    data = pd.merge(data, feat, on=keys, how='left')

    # 4. 商家发行不同种类的优惠券
    keys = ['Merchant_id']
    feat = data.groupby(keys)['Coupon_id'].nunique().reset_index()
    feat.rename(columns={'Coupon_id': 'label_field_Merchant_differ_Coupon_received_cnt'}, inplace=True)
    data = pd.merge(data, feat, on=keys, how='left')

    #新增
    # 5. 商家忠诚度（被同一用户领取的次数/被领取总次数） 
    data['User_Merchant_loyalty'] = data['label_field_User_Merchant_received_cnt'] / (data['label_field_User_received_cnt'] + 1)

  #  data['Merchant_Coupon_popularity_rank'] = data.groupby('Merchant_id')['label_field_Merchant_Coupon_received_cnt'].rank(ascending=False)


    #0.8049新增特征



    data.drop(['cnt'], axis=1, inplace=True)

    return data



def get_User_Merchant_feature(label_field):

    data = label_field.copy()
    data['cnt'] = 1

    '''
    # 1. 用户在该商家领券占用户总领券的比例
    data['label_User_Merchant_receive_rate'] = data['label_field_User_Merchant_received_cnt'] / (
                data['label_field_User_received_cnt'] + 1)

    # 2. 用户对该商家的偏好度（领券数/商家总被领数）
    data['label_User_Merchant_preference'] = data['label_field_User_Merchant_received_cnt'] / (
                data['label_field_Merchant_received_cnt'] + 1)

    
    # 3. 用户在该商家领取的不同优惠券种类数
    keys = ['User_id', 'Merchant_id']
    feat = data.groupby(keys)['Coupon_id'].nunique().reset_index()
    feat.rename(columns={'Coupon_id': 'label_User_Merchant_differ_Coupon_cnt'}, inplace=True)
    data = pd.merge(data, feat, on=keys, how='left')
    data['label_User_Merchant_differ_Coupon_cnt'].fillna(0, inplace=True)
    '''
    # 4.用户领取该商家优惠券的平均折扣
    keys = ['User_id', 'Merchant_id']
    feat = data.groupby(keys)['discount_rate'].mean().reset_index()
    feat.rename(columns={'discount_rate': 'label_User_Merchant_avg_discount'}, inplace=True)
    data = pd.merge(data, feat, on=keys, how='left')


    data.drop(['cnt'], axis=1, inplace=True)
    return data

'''
def get_Merchant_Coupon_feature(label_field):

    data = label_field.copy()
    data['cnt'] = 1

    # 1. 商家发行该优惠券占商家总发行的比例
    data['Merchant_Coupon_popularity'] = data['label_field_Merchant_Coupon_received_cnt'] / (data['label_field_Merchant_received_cnt'] + 1)

   
    # 2. 该券的用户集中度（被多少用户领取/被领取次数）
    keys = ['Merchant_id', 'Coupon_id']
    feat = data.groupby(keys)['User_id'].nunique().reset_index()
    feat.rename(columns={'User_id': 'Merchant_Coupon_user_cnt'}, inplace=True)
    data = pd.merge(data, feat, on=keys, how='left')
    data['Merchant_Coupon_user_concentration'] = data['Merchant_Coupon_user_cnt'] / (data['label_field_Merchant_Coupon_received_cnt'] + 1)
    # 3. 该券的发放天数
    keys = ['Merchant_id', 'Coupon_id']
    feat = data.groupby(keys)['Date_received'].nunique().reset_index()
    feat.rename(columns={'Date_received': 'Merchant_Coupon_active_days'}, inplace=True)
    data = pd.merge(data, feat, on=keys, how='left')
    
    # 4. 该券的日均被领取次数
    data['Merchant_Coupon_daily_receive'] = data['label_field_Merchant_Coupon_received_cnt'] / (data['Merchant_Coupon_active_days'] + 1)

    # 5. 该券是否是商家的主推券（被领取次数是否超过平均）
    keys = ['Merchant_id']
    feat = data.groupby(keys)['label_field_Merchant_Coupon_received_cnt'].mean().reset_index()
    feat.rename(columns={'label_field_Merchant_Coupon_received_cnt': 'Merchant_avg_Coupon_receive'}, inplace=True)
    data = pd.merge(data, feat, on=keys, how='left')
    data['is_main_coupon'] = (data['label_field_Merchant_Coupon_received_cnt'] > data['Merchant_avg_Coupon_receive']).astype(int)

    data.drop(['cnt', 'Merchant_Coupon_user_cnt', 'Merchant_avg_Coupon_receive'], axis=1, inplace=True)
    
    return data
'''


def get_label_field_feature(label_field):
    """
    整合用户 + 商家特征
    """
    data = label_field.copy()

    data = get_label_user_feature(data)

    data = get_label_Merchant_feature(data)

#    data = get_User_Merchant_feature(data)

    return data
