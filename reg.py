import datetime
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

var = []
var2 = []
data_save = '/Users/lvfreud/Documents/python/factor_data/'
output = '/Users/lvfreud/Desktop/'


def roll_reg(roll_data):  # rolling_regress
    dm = roll_data['ym'].max()
    x = roll_data[var]
    x = sm.add_constant(x)
    y = roll_data['r_f']
    model = sm.OLS(y, x, missing='drop')
    results = model.fit()
    beta, p, r2 = results.params, results.pvalues, results.rsquared
    dp = roll_data.drop(roll_data[roll_data['ym'] < dm].index)
    if len(dp) <= 1:
        dp['r_p'] = -1
        return dp, p, r2
    dp['r_p'] = np.dot(sm.add_constant(dp[var2]), beta)
    return dp, p, r2


def main(gn, con_str, roll_days, stock_num, main_var):
    global var, var2
    var = main_var.copy()
    var.extend(['inc', 'ebi'])
    var2 = main_var.copy()
    var2.extend(['incs', 'ebis'])
    data = pd.read_csv(data_save + 'su.csv')
    data['ym'] = pd.to_datetime(data['ym'])

    # WinD
    concept = pd.read_excel(data_save + 'concept.xlsx', sheet_name=0)
    concept['Stkcd'] = concept['证券代码'].str[:6].astype('int')
    if gn:
        concept['cp'] = concept['行业'].str.contains(con_str)
    else:
        concept['cp'] = concept['所属概念板块'].str.contains(con_str)

    # iFinD
    # concept = pd.read_csv(data_save + '概念.csv')
    # concept['Stkcd'] = concept['股票代码'].str[:6].astype('int')
    # concept['cp'] = concept['所属概念'].str.contains(con_str)

    data = data.merge(concept[['Stkcd', 'cp', '证券简称']], how='left', on='Stkcd')
    data = data[data['cp'] == 1]  # 筛选行业
    data = data[~data['证券简称'].str.contains('ST')]
    data = data.fillna(0)
    date = data['ym']
    date = date.drop_duplicates()
    y_sum = pd.DataFrame()
    p_sum = pd.DataFrame()
    r2_sum = []
    delta = datetime.timedelta(days=roll_days)
    date = date[date < data['ym'].max() - datetime.timedelta(days=roll_days - 30)]
    for d in date:
        rdata = data.drop(data[(data['ym'] > d + delta) | (data['ym'] < d)].index)
        y_pz, pz, r2z = roll_reg(rdata)
        if len(y_sum) == 0:
            y_sum = y_pz
            p_sum = pz
        else:
            y_sum = pd.concat([y_sum, y_pz], axis=0)
            p_sum = pd.concat([p_sum, pz], axis=1)
        r2_sum.append(r2z)
    st = data.groupby(['Stkcd']).min()
    st = st['ym']
    st = st.rename('st')
    y_sum = y_sum.merge(st, how='left', on='Stkcd')
    y_sum['stm'] = y_sum['ym'] - y_sum['st']
    y_sum = y_sum.drop(y_sum[y_sum[var].duplicated()].index)
    de = datetime.timedelta(days=300)
    y_sum = y_sum[y_sum['stm'] > de]  # 筛选成立300天的企业
    select_arr = y_sum.sort_values(by=['r_p'], ascending=[False]).reset_index(drop=True)
    select_arr = select_arr.groupby(['ym']).head(stock_num)
    select_arr = select_arr.sort_values(by=['ym'], ascending=[True]).reset_index(drop=True)  # 选股序列
    da = select_arr['ym'].drop_duplicates()  # 获取时间序列
    select_arr['w'] = np.zeros(len(select_arr))
    for i in da:
        u = select_arr[select_arr['ym'] == i]['r_p']
        idd = select_arr[select_arr['ym'] == i]['Stkcd']
        idd = idd.reset_index(drop=True)
        cor = data[data['Stkcd'] == idd[0]][['ym', 'r_su']]
        for j in idd:  # 获取收益率矩阵
            if j == idd[0]:
                continue
            m = data[data['Stkcd'] == j][['ym', 'r_su']]
            m = m.rename(columns={'r_su': 'r_su_' + str(j)})
            cor = cor.merge(m, on=['ym'], how='outer')
        cor = cor.fillna(0)
        co = cor[(cor['ym'] <= i) & (cor['ym'] > i - delta)].iloc[:, 1:].corr()
        w = np.dot(np.linalg.inv(co), u)  # 根据收益率相关性以及效用函数设定，求各股权重
        w = w * (w > 0)
        if w.sum() != 0:
            w = w / w.sum()
        select_arr.loc[select_arr['ym'] == i, 'w'] = w
    select_arr['r'] = select_arr['r_f'] * select_arr['w']
    syl = select_arr[['ym', 'r']].groupby(['ym']).sum()
    syl = syl.cumsum()
    results = select_arr[['Stkcd', 'w']].iloc[-stock_num:, :]
    results['板块'] = con_str
    results['r'] = syl.iloc[-1].values[0]
    # 绘图略
    # syl.plot(figsize=(18, 8), ylim=[0, 6])
    # plt.savefig('si.png')
    # plt.show()
    return results


if __name__ == '__main__':
    re = [main(0, '芯片', 200, 4, ['to_sd', 'ret_m', 'r_sd', 'r_m', 'lnd_m']),
          main(0, '数字经济', 200, 4, ['to_sd', 'ret_m', 'r_sd', 'r_v', 'lnd_m']),
          main(0, '传媒', 400, 4, ['to_sd', 'ret_m', 'r_sd', 'r_su', 'lnd_m']),
          main(0, '钠离子电池', 300, 4, ['to_sd', 'ret_m', 'r_sd', 'r_su', 'lnd_m']),
          main(1, '半导体', 200, 4, ['to_sd', 'ret_m', 'r_m', 'r_v', 'lnd_m']),
          main(1, '酒类', 200, 4, ['to_sd', 'ret_m', 'r_v', 'lnd_m']),
          main(1, '医疗保健', 200, 4, ['to_sd', 'ret_m', 'r_v', 'lnd_m']),
          main(1, '航空与物流', 200, 4, ['to_sd', 'ret_m', 'r_v', 'lnd_m']),
          main(1, '零售', 200, 4, ['to_sd', 'ret_m', 'r_v', 'lnd_m']),
          main(1, '化肥农药', 200, 4, ['to_sd', 'ret_m', 'r_v', 'lnd_m']),
          main(1, '通信设备', 200, 4, ['to_sd', 'ret_m', 'r_v', 'lnd_m']),
          main(1, '多元金融', 200, 4, ['to_sd', 'ret_m', 'r_sd', 'r_v', 'lnd_m']),
          ]
    re = pd.concat(re)
    re.to_excel(output + 'select.xlsx', index=False)
