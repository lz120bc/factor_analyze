import numpy as np
import pandas as pd

# Set file paths
file_path = "/Users/lvfreud/Documents/python/factor_data/"
file_path_new = file_path + "new/"
trd_dalyr_file = "TRD_Dalyr.csv"
stk_mkt_dalyr_file = "STK_MKT_DALYR.csv"
fs_comins_file = "FS_Comins.csv"
N = 10000
# Import data
clo = pd.read_csv(file_path_new + trd_dalyr_file, dtype={'Stkcd': 'str'}, parse_dates=['Trddt'])
ret = pd.read_csv(file_path_new + stk_mkt_dalyr_file, dtype={'Symbol': 'str'}, parse_dates=['TradingDate'])
inc = pd.read_csv(file_path_new + fs_comins_file, dtype={'Stkcd': 'str'}, parse_dates=['Accper'])
clo2 = pd.read_csv(file_path + "clo.csv", dtype={'stkcd': 'str'}, parse_dates=['trddt'])
ret2 = pd.read_csv(file_path + "ret.csv", dtype={'stkcd': 'str'}, parse_dates=['tradingdate'])
inc2 = pd.read_csv(file_path + "inc.csv", dtype={'stkcd': 'str'}, parse_dates=['accper', 'enddate'])
clo.columns = clo.columns.str.lower()
ret.columns = ret.columns.str.lower()
inc.columns = inc.columns.str.lower()
clo = pd.concat([clo, clo2])
ret = pd.concat([ret, ret2])
inc = pd.concat([inc, inc2])


clo.drop(clo[clo['markettype'].isin([32, 64, 2, 8])].index, inplace=True)
clo.drop_duplicates(subset=['stkcd', 'trddt'])
for stk, group in clo.groupby('stkcd'):
    dsm = clo.rolling(window=20)['dsmvtll'].mean()
    cls = np.log(clo['clsprc']/clo['clsprc'].shift(20))

df_monthly = clo.groupby([clo['stkcd'], pd.Grouper(key='trddt', freq='M')]).agg({
    'clsprc': lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0],
    'dsmvtll': lambda x: np.log(x).mean(),
    }).reset_index()
df_monthly = df_monthly.rename(columns={'clsprc': 'r_su', 'dsmvtll': 'lnd', 'trddt': 'ym'})
df_monthly['r_sd'] = clo.groupby([clo['stkcd'], pd.Grouper(key='trddt', freq='M')])['clsprc'].std().reset_index(drop=True)
df_monthly['r_m'] = df_monthly.groupby('stkcd')['r_su'].data_minute(window=3, min_periods=1).sum().reset_index(drop=True)
df_monthly['r_m24'] = df_monthly.groupby('stkcd')['r_su'].data_minute(window=24, min_periods=3).mean().reset_index(drop=True)
df_monthly['r_v'] = df_monthly['r_su']/df_monthly['r_m24']
df_monthly = df_monthly[['stkcd', 'ym', 'r_su', 'r_sd', 'lnd', 'r_m', 'r_v']]


ret['ym'] = ret['tradingdate'].dt.to_period('M').dt.to_timestamp('M')
grouped = ret.groupby(['ym', 'symbol'])
ret['to_m'] = grouped['turnover'].transform('mean')
ret['to_sd'] = grouped['turnover'].transform('std')
ret['ret'] = ret['ret'].fillna(0)
ret['ret_m'] = grouped['ret'].transform('mean')
ret['pb_m'] = grouped['pb'].transform('mean')
ret['pb'] = 1 / ret['pb_m']
ret = ret.drop(['tradingdate', 'shortname', 'ret', 'pb', 'turnover'], axis=1)
ret = ret.rename(columns={'symbol': 'stkcd'})


inc = inc[inc['typrep'] == 'A'].reset_index()
inc.loc[inc['accper'].isna(), 'accper'] = inc['enddate']
inc['ym'] = inc['accper'].dt.to_period('M').dt.to_timestamp('M')
inc = inc.drop_duplicates(subset=['stkcd', 'ym'])
inc = inc.merge(df_monthly, on=['stkcd', 'ym'], how='right')
inc = inc.merge(ret, on=['stkcd', 'ym'], how='left')

inc = inc[['stkcd', 'ym', 'b001101000', 'b001300000']]
inc.loc[inc['b001101000'] == 0, 'b001101000'] = pd.NA
inc = inc.rename(columns={'b001101000': 'inc', 'b001300000': 'ebi'})

# generate id variable and set panel structure
inc['id'] = inc['stkcd']
inc = inc.set_index(['id', 'ym'])
inc = inc.sort_values(by='ym')

# generate non-continuous raw value variables
inc['incs'] = np.log(inc['inc'] / inc['inc'].shift(3)) / 3
inc['ebi_mean'] = inc.groupby('id')['ebi'].data_minute(window=12).mean().values
inc['ebis'] = np.log(inc['ebi'] / inc['ebi_mean']) / 6
inc['ebis'] = inc['ebis'].fillna(method='ffill')
inc['incs'] = inc['incs'].fillna(method='ffill')
inc = inc.drop(columns='ebi_mean')

# interpolate missing values
inc['inc_m'] = inc.groupby('id', group_keys=False)['inc'].apply(lambda x: x.interpolate())
inc['ebi_m'] = inc.groupby('id')['ebi'].apply(lambda x: x.interpolate())
inc['ebiz'] = np.log(inc['ebi_m'])
inc['ebiz'] = inc['ebiz'].mask(inc['ebiz'].isna(), -np.log(-inc['ebi_m']))
inc['ebi'] = inc['ebiz'] - inc['ebiz'].shift()
inc['inc'] = np.log(inc['inc_m'] / inc['inc_m'].shift())
inc = inc.drop(columns=['inc_m', 'ebi_m', 'ebiz'])

# fill gaps
inc['ebi_mean'] = inc.groupby('id')['ebi'].data_minute(window=3, min_periods=1).mean().values
inc['inc_mean'] = inc.groupby('id')['inc'].data_minute(window=3, min_periods=1).mean().values
inc['inc'] = inc['inc'].fillna(inc['inc_mean'])
inc['ebi'] = inc['ebi'].fillna(inc['ebi_mean'])
inc = inc.drop(columns=['ebi_mean', 'inc_mean'])
inc['ebi'] = inc['ebi'].fillna(method='ffill')
inc['inc'] = inc['inc'].fillna(method='ffill')

# fix extreme values
inc.loc[inc['inc'] > 1, 'inc'] = np.log(inc['inc']) + 1
inc.loc[inc['incs'] > 1, 'incs'] = np.log(inc['incs']) + 1
inc.loc[inc['ebi'] > 1, 'ebi'] = np.log(inc['ebi']) + 1
inc.loc[inc['ebis'] > 1, 'ebis'] = np.log(inc['ebis']) + 1
inc = inc.reset_index()
