# %% import packages to use
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import py7zr
import os

figure_path = os.path.join(os.getcwd(), 'figures')
table_path = os.path.join(os.getcwd(), 'tables')
idx = pd.IndexSlice

# %% un-package the 7z directory and read the data
with py7zr.SevenZipFile(os.path.join(os.getcwd(), 'data', 'full202052.7z'), mode='r') as z:
    z.extractall()

# read the dataset
df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'full202052.dat'))

# %% exclude unimportant rows and columns to save
# exclude rows with total value by type of trade flow
# exclude rows with TRADE_TYPE info that is neither Extra-EU nor Intra-EU
exclude_rows = ((df.PRODUCT_NC == 'TOTAL') |
                (df.PRODUCT_CPA2002 == 'TOTAL') |
                (df.PRODUCT_CPA2008 == 'TOTAL') |
                (df.PRODUCT_CPA2_1 == 'TOTAL') |
                (df.PRODUCT_BEC == 'TOTAL') |
                (df.PRODUCT_BEC == 'TOTAL') |
                (df.TRADE_TYPE == 'K'))

# exclude columns not used in the current analysis
exclude_columns = [
    'PERIOD',           # same value of
    'SUPP_UNIT',        # not used in current analysis
    'SUP_QUANTITY',     # not used in current analysis
    'QUANTITY_IN_KG',   # not used in current analysis
    'STAT_REGIME',      # not used in current analysis
    'PRODUCT_CPA2002',  # not used in current analysis
    'PRODUCT_CPA2008',  # not used in current analysis
    'PRODUCT_CPA2_1'    # not used in current analysis
]

df = df.loc[(~exclude_rows), (~df.columns.isin(exclude_columns))]
df = (df.replace({'FLOW': {1: 'imports', 2: 'exports'}})
      .replace({'TRADE_TYPE': {"I": 'Intra-EU', 'E': 'Extra-EU'}}))

# %% (1) Monthly total EU imports and exports
units = 1e+012  # rade volume in billion euros
eu_flow = df.groupby(['FLOW', 'TRADE_TYPE'])['VALUE_IN_EUROS'].sum() / units

# plotting
sns.set_style("whitegrid")
fig = plt.figure(constrained_layout=True)
specs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
ax1 = fig.add_subplot(specs[:-1, :])
ax2 = fig.add_subplot(specs[1, 0])
ax3 = fig.add_subplot(specs[1, 1])

# volume and share of imports and exports by trade type
eu_flow.unstack().plot.bar(ax=ax1, stacked=True)
ax1.set_ylabel('Total EU flow (trillion €)')
ax1.set_xlabel(None)
ax1.legend(bbox_to_anchor=(0.6, -0.1), ncol=1)

eu_flow.unstack().loc['exports'].plot.pie(ax=ax2,
                                          legend=False,
                                          labels=None,
                                          autopct='%1.1f%%'
                                          )
eu_flow.unstack().loc['imports'].plot.pie(ax=ax3,
                                          legend=False,
                                          labels=None,
                                          autopct='%1.1f%%'
                                          )
ax2.set_ylabel(None)
ax3.set_ylabel(None)
fig.suptitle('Volume and share of EU imports and exports by trade type')
fig_name = "_".join("volume and share of EU imports and exports by trade type".split())
fig.savefig(os.path.join(figure_path, fig_name + ".png"))

# %% share of import & exports by trade type
shares = pd.pivot_table(df,
                        values='VALUE_IN_EUROS',
                        index=['DECLARANT_ISO', 'TRADE_TYPE'],
                        columns=['FLOW'],
                        aggfunc=np.sum)

# contribution of EU Member States to intra- and extra-EU trade
shares = (
    shares.groupby(pd.Grouper(level='TRADE_TYPE'))
          .transform(lambda x: 100 * x / x.sum())
)

# %%
sns.set_style('whitegrid')
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
sns.set_color_codes('muted')
extra_data = shares.loc[idx[:, 'Extra-EU'], :].reset_index().set_index('DECLARANT_ISO')
extra_data = extra_data.nlargest(5, 'exports').sort_values(by='exports', ascending=False)
extra_data.plot.bar(ax=ax1)
ax1.set_title('Contribution of each member to extra-EU trade \n (top 5 exporters)')
ax1.set_xlabel(None)
ax1.set_ylabel("Percent")
ax1.legend(ncol=1, loc='upper right')

intra_data = shares.loc[idx[:, 'Extra-EU'], :].reset_index().set_index('DECLARANT_ISO')
intra_data = intra_data.nlargest(5, 'exports').sort_values(by='exports', ascending=False)
intra_data.plot.bar(ax=ax2)
ax2.set_title('Contribution of each member to intra-EU trade \n (top 5 exporters)')
ax2.set_xlabel(None)
ax2.legend(ncol=1, loc='upper right')
sns.despine(left=True, top=True)
fig.tight_layout()

fig_name = "_".join("Contribution of EU Member States to intra-EU and extra-EU trade".split())
fig.savefig(os.path.join(figure_path, fig_name + ".png"))


# %% Number of products imported & exported by member country and trade type
def number_distinct_values(x: pd.Series, **kwargs) -> pd.Series:
    return x.nunique(**kwargs)


# number of distinct products according to SITE code: by state and flow and trade type
unique = pd.pivot_table(df,
                        values='PRODUCT_SITC',
                        index=['DECLARANT_ISO', 'TRADE_TYPE'],
                        columns=['FLOW'],
                        aggfunc=number_distinct_values)

# number of distinct products according to SITE code: for EU by flow type
eu_total = pd.pivot_table(df,
                          values='PRODUCT_SITC',
                          columns=['FLOW'],
                          aggfunc=number_distinct_values)

# percent from overall EU
unique = 100 * unique.div(eu_total.T.iloc[:, 0])

# 5 member states by diversity of exports and imports
exports = unique.unstack()['exports'].nlargest(5, 'Extra-EU')
imports = unique.unstack()['imports'].nlargest(5, 'Extra-EU')


# %%
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
exports.sort_index().plot.bar(ax=ax1)
imports.sort_index().plot.bar(ax=ax2)
ax1.set_xlabel('Top 5 exporters')
ax2.set_xlabel('Top 5 importers')
ax1.set_ylabel(None)
plt.xticks(rotation=0)
ax1.legend(ncol=1, loc='upper right')
ax2.legend(ncol=1, loc='upper right')
sns.despine(left=True, top=True)
fig.suptitle("Percent of the number of imported and exported products \n by Member State from the EU total")
fig.tight_layout()
fig_name = "_".join("Share of the number of imported and exported products from the EU total".split())
fig.savefig(os.path.join(figure_path, fig_name + ".png"))

# %% TODo Trade in goods by top 5 partners
#  who are the top 5 exporters to importers from the EU
partners = pd.pivot_table(df.loc[(df.TRADE_TYPE == 'Extra-EU'), :],
                          values='VALUE_IN_EUROS',
                          index=['PARTNER_ISO'],
                          columns=['FLOW'],
                          aggfunc=np.sum)

# export destination
# destinations = [x for x in df.PARTNER_ISO.unique() if x not in df.DECLARANT_ISO.unique()]
# dst_data = partners.loc[partners.index.isin(destinations), :]
#
# partners = partners.apply(lambda x: 100 * x / x.sum())

# %% Contribution of EU Member States to extra-EU trade - seasonally adjusted data bn €
