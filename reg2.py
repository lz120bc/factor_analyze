import datetime
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

var = []
var2 = []
data_save = '/Users/lvfreud/Documents/python/factor_data/'
output = '/Users/lvfreud/Desktop/'


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
    results = []
    for _, groups in data.groupby('ym'):
        groups['select'] = groups[var].sum()
        ss = groups.sort_values('select').head(stock_num)
        results.append(ss)
    results = pd.concat(results)
    rr = results.groupby('ym')['r_f'].mean().reset_index()
    # 绘图略
    # syl.plot(figsize=(18, 8), ylim=[0, 6])
    # plt.savefig('si.png')
    # plt.show()
    return results, rr


if __name__ == '__main__':
    re, y = main(0, '芯片', 200, 4, ['to_sd', 'ret_m', 'r_sd', 'r_m', 'lnd_m'])
