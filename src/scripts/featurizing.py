# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np 
import pandas as pd 
#import modin.pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# %%
orig_df = pd.read_csv('../../data/data/datastorm_policy_data.csv')

# %%
orig_df.head()
print(orig_df.info())
print(orig_df.isnull().sum())
# %%
print(orig_df.columns)
# %%
def month_rank(df):
    df['policy_snapshot'] = pd.to_datetime(df['policy_snapshot_as_on'],format='%Y%m%d')
    df['month_rank'] = df['policy_snapshot'].dt.year*12+df['policy_snapshot'].dt.month
    df['month_rank'] = df['month_rank'] - df['month_rank'].min()
    return df
# %%
up1_df = month_rank(orig_df)

# %%
def get_num_of_policies(df):
    n = df["policy_code"].nunique()
    return n

# %%
gp_df = up1_df.groupby(by=["month_rank","client_code"]).apply(get_num_of_policies).reset_index()

# %%
gp_df.rename(columns={0:'num_pl'},inplace=True)
gp_df.head()

# %%
up1_df = pd.merge(up1_df, gp_df, on=["month_rank","client_code"])
# %%
up1_df.head()

# %%
def get_target(df):
    if df.shape[0] > 2:
        df_c = df.copy()
        sr_df = df_c.sort_values(by=["month_rank"])
        this_mon = np.delete(sr_df['num_pl'].values,0)
        next_mon = np.append(np.delete(sr_df['num_pl'].values,[0,1]),sr_df['num_pl'].values[-1])
        res_df = df_c.iloc[1:, :]
        res_df['is_added'] = next_mon-this_mon
        return res_df
    else:
        return 
# %%
gp1_df = up1_df.groupby(by=["client_code"]).apply(get_target)
# %%
gp1_df['is_added']
# %%
sns.histplot(data=gp1_df['is_added'])
# %%
def binarize(x):
    if x>0:
        return 1 
    else:
        return 0
# %%
gp1_df['is_cross_sell'] = gp1_df['is_added'].apply(binarize)
# %%
sns.histplot(data=gp1_df['is_cross_sell'])
# %%
gp1_df.to_csv('filtered.csv', index=False)
# %%
