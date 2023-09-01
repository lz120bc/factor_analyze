import matplotlib.pyplot as plt
import pandas as pd

var = []
var2 = []
data_save = '/Users/lvfreud/Documents/python/factor_data/'
output = '/Users/lvfreud/Desktop/'
data = pd.read_csv(data_save + 'su.csv')

factor_ic = []
for ym, group in data.groupby(['ym']):
    group[['to_sd_ic', 'ret_m_ic', 'r_sd_ic', 'r_m_ic', 'lnd_m_ic', 'inc_ic', 'ebi_ic']] = group[
        ['to_sd', 'ret_m', 'r_sd', 'r_m', 'lnd_m', 'inc', 'ebi']].apply(lambda x: x.corr(group['r_f']))
    group[['to_sd_ric', 'ret_m_ric', 'r_sd_ric', 'r_m_ric', 'lnd_m_ric', 'inc_ric', 'ebi_ric']] = group[
        ['to_sd', 'ret_m', 'r_sd', 'r_m', 'lnd_m', 'inc', 'ebi']].apply(lambda x: x.rank().corr(group['r_f'].rank()))
    factor_ic.append(group)
rolling_data = pd.concat(factor_ic)

rolling_data.groupby('ym')[['to_sd_ic', 'ret_m_ic', 'r_sd_ic', 'r_m_ic', 'lnd_m_ic', 'inc_ic', 'ebi_ic']].mean().plot()
rolling_data.groupby('ym')[['to_sd_ric', 'ret_m_ric', 'r_sd_ric', 'r_m_ric', 'lnd_m_ric', 'inc_ric', 'ebi_ric']].mean().plot()
plt.show()

# rolling_data[['to_sd', 'ret_m', 'r_sd', 'r_m', 'lnd_m']].corr()
rolling_data[['to_sd_ic', 'ret_m_ic', 'r_sd_ic', 'r_m_ic', 'lnd_m_ic', 'inc_ic', 'ebi_ic']].mean()
