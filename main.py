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

params = {'legend.fontsize': 'medium',
          'figure.titlesize': 'x-large',
          'axes.labelsize': 'small',
          'axes.titlesize': 'large',
          'xtick.labelsize': 'small',
          'ytick.labelsize': 'small',
          "font.family": "Times New Roman"
          }
plt.rcParams.update(params)

# %% un-package the 7z directory and read the data
with py7zr.SevenZipFile(os.path.join(os.getcwd(), 'data', 'full202052.7z'), mode='r') as z:
    z.extractall()

# read the dataset
df = pd.read_csv(os.path.join(os.getcwd(), 'full202052.dat'))

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

# %% exporting results of trade volume to a table
table = eu_flow.to_frame('trillion').unstack().round(3)
table.columns = [f"{b} ({a} €)" for a, b in table.columns]
tab_name = "_".join("volume and share of EU imports and exports by trade type".split())
table.to_csv(os.path.join(table_path, tab_name + ".csv"))

# %% prepare the graph and export to figures
sns.set_style("whitegrid")
fig = plt.figure(constrained_layout=True)
specs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
ax1 = fig.add_subplot(specs[:-1, :])
ax2 = fig.add_subplot(specs[1, 0])
ax3 = fig.add_subplot(specs[1, 1])

# volume and share of imports and exports by trade type
eu_flow.unstack().plot.bar(ax=ax1, stacked=True)
ax1.set_ylabel('EU trade flows (trillion €)')
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

# %% export result in csv table
tab_name = "_".join("Share of imports and exports by member state and trade type".split())
(
    shares.unstack()
    .sort_values(by=('exports', 'Intra-EU'), ascending=False)
    .to_csv(os.path.join(table_path, tab_name + ".csv"))
)

# %%
sns.set_style('whitegrid')
fig, (ax1, ax2) = plt.subplots(figsize=(10, 6), ncols=2, sharey=True)
sns.set_color_codes('muted')
extra_data = shares.loc[idx[:, 'Extra-EU'], :].reset_index().set_index('DECLARANT_ISO')
extra_data = extra_data.nlargest(5, 'exports').sort_values(by='exports', ascending=False)
extra_data.plot.bar(ax=ax1)
ax1.set_title('Contribution of Member States to Extra-EU trade')
ax1.set_xlabel("Top 5 exporting Member States")
ax1.set_ylabel("Percent")
ax1.legend(ncol=1, loc='upper right')

intra_data = shares.loc[idx[:, 'Extra-EU'], :].reset_index().set_index('DECLARANT_ISO')
intra_data = intra_data.nlargest(5, 'exports').sort_values(by='exports', ascending=False)
intra_data.plot.bar(ax=ax2)
ax2.set_title('Contribution of Member States to Intra-EU trade')
ax2.set_xlabel('Top 5 exporting Member States')
ax2.legend(ncol=1, loc='upper right')
sns.despine(right=True, top=True)
fig.tight_layout()

fig_name = "_".join("Contribution of EU Member States to intra-EU and extra-EU trade".split())
fig.savefig(os.path.join(figure_path, fig_name + ".png"))


# %% Number of products imported & exported by member country and trade type
def number_distinct_values(x: pd.Series, **kwargs) -> pd.Series:
    """Method to compute the number of distinct values in a pandas series.

    Parameters
    ----------
    x : pd.Series
        A Series for whom we intend to find the number of distinct values.
    kwargs : Any
        Keyword arguments to pandas' nunique method on Series.

    Returns
    -------
    int
        Number of distinct values in x.
    """
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

# %% export results in csv document
table = unique.unstack().round(2)
table.index.name = None
tab_name = "_".join("Share from EU total of the number of traded goods by Member States".split())
table.to_csv(os.path.join(table_path, tab_name + ".csv"))

# %%
sns.set_style('whitegrid')
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
exports.sort_index().plot.bar(ax=ax1)
imports.sort_index().plot.bar(ax=ax2)
ax1.set_xlabel('Top 5 exporters')
ax2.set_xlabel('Top 5 importers')
ax1.set_ylabel(None)
plt.xticks(rotation=0)
ax1.legend(ncol=1, loc='upper right')
ax2.legend(ncol=1, loc='upper right')
sns.despine(right=True, top=True)
fig.suptitle("Percent of the number of imported and exported products \n by Member State from the EU total")
fig.tight_layout()
fig_name = "_".join("Share of the number of imported and exported products from the EU total".split())
fig.savefig(os.path.join(figure_path, fig_name + ".png"))

# %% EU's major trading partners
partners = pd.pivot_table(df.loc[(df.TRADE_TYPE == 'Extra-EU'), :],
                          values='VALUE_IN_EUROS',
                          index=['PARTNER_ISO'],
                          columns=['FLOW'],
                          aggfunc=np.sum)

# third countries
third_nations = list(set(df.PARTNER_ISO).difference(set(df.DECLARANT_ISO)))

# main export destinations
partner_share = (
    partners.loc[partners.index.isin(third_nations), :]
    .apply(lambda x: 100 * x / x.sum())
)
export_dst = partner_share.nlargest(5, 'exports').loc[:, 'exports']
export_dst.loc['Others'] = 100 - export_dst.sum()
#
# partners = partners.apply(lambda x: 100 * x / x.sum())
# main origins for imports
import_origins = partner_share.nlargest(5, 'imports').loc[:, 'imports']
import_origins.loc['Others'] = 100 - import_origins.sum()
# %% export table
tab_name = "_".join("Major trading partners of the EU".split())
(partner_share.sort_values(by=['exports', 'imports'], ascending=False)
 .to_csv(os.path.join(table_path, tab_name + ".csv"))
 )

# %% Contribution of EU Member States to extra-EU and intra-EU trade
sns.color_palette("Set2")
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 8))
export_dst.plot.pie(ax=ax1,
                    label="Major destinations of EU exports",
                    autopct='%1.1f%%',
                    shadow=True,
                    startangle=90)
sns.color_palette("tab10")
import_origins.plot.pie(ax=ax2,
                        label="Major origins of EU imports",
                        autopct='%1.1f%%',
                        shadow=True,
                        startangle=90)

ax1.set_ylabel(None)
ax2.set_ylabel(None)
ax1.set_title("Main destinations of EU exports")
ax2.set_title("Main origins of EU imports")
fig.suptitle("EU's main trading partners and their shares in total Extra-EU trade")
fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig_name = "_".join("Major trading partners and their shares in Extra-EU".split())
fig.savefig(os.path.join(figure_path, fig_name + ".png"))


# %% PRODUCT CATEGORY BASED ON PRODUCT_BEC:
def map_product_category(x: str, category_dict: dict[str, tuple]):
    """Method to map a product identifier to its group name.

    Parameters
    ----------
    x : str
        A string identifying BEC4 product classification.
    category_dict : dict[str, tuple]
        A dict of category name as a key and a
        tuple of product identifiers as a value.

    Returns
    -------
    str
        Category name for the product identifier.
    """
    for kind, elm in category_dict.items():
        if x.startswith(elm):
            return kind
    return np.nan


product_types = {
    'Intermediate goods': ('111', '121', '21', '22', '31', '322', '42', '53'),
    'Consumption goods': ('112', '122', '522', '61', '62', '63'),
    'Capital goods': ('41', '521'),
}

# consumption goods types
durability = {
    "Durable": ('61',),
    "Semi-durable": ('62',),
    "Non-durable": ('63',)
}

df["product_type"] = df.PRODUCT_BEC.apply(map_product_category,
                                          category_dict=product_types)
df["durability"] = df.PRODUCT_BEC.apply(map_product_category,
                                        category_dict=durability)

prod_type_df = pd.pivot_table(df,
                              values='VALUE_IN_EUROS',
                              index=['DECLARANT_ISO', 'TRADE_TYPE', 'product_type'],
                              columns=['FLOW'],
                              aggfunc=np.sum)

dur_type_df = pd.pivot_table(df,
                             values='VALUE_IN_EUROS',
                             index=['DECLARANT_ISO', 'TRADE_TYPE', 'durability'],
                             columns=['FLOW'],
                             aggfunc=np.sum)

# %%
# Share of EU trade flow by trade type & product category
share_prod_type_eu = (
    (prod_type_df.groupby([pd.Grouper(level='TRADE_TYPE'),
                           pd.Grouper(level='product_type')])
     .sum()
     )
     .div(prod_type_df.sum())
     .mul(100)
     .round(2)
)
tab_name = "_".join("Share of EU trade flow by trade type and product category".split())
share_prod_type_eu.to_csv(os.path.join(table_path, tab_name + ".csv"))

# Contribution of MS to EU's total trade flow by trade type & product category
share_prod_type_members = (
    prod_type_df.unstack(level=['TRADE_TYPE', 'product_type'])
        .apply(lambda x: 100 * x / x.sum())
)
tab_name = "_".join("Contribution of MS to EU's total trade flow by trade type and product category".split())
share_prod_type_members.to_csv(os.path.join(table_path, tab_name + ".csv"))

# Contribution of product category in EU's total trade flow by trade type
prod_type_df1 = prod_type_df.unstack(level='TRADE_TYPE')
share_prod_type_mbr = (
    prod_type_df1.div(prod_type_df1.sum(0))
        .mul(100)
        .round(3)
)
tab_name = "_".join("Contribution of product category in total EU trade flow by trade type".split())
share_prod_type_mbr.to_csv(os.path.join(table_path, tab_name + ".csv"))


# %% EU trade flows by BEC product category and trade type
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
for f, flow in enumerate(("exports", "imports")):
    title_text = "EU trade flows by BEC product category and trade type"
    for c, trade in enumerate(('Extra-EU', 'Intra-EU')):
        tmp = share_prod_type_eu[flow].loc[trade]
        tmp.plot.pie(ax=ax[f, c], autopct='%1.1f%%', shadow=True, startangle=90)
        ax[f, c].set_title(f"{trade} {flow}")
        ax[f, c].set_ylabel(None)
    fig.suptitle(title_text)
    fig.tight_layout()
    fig_name = "_".join(title_text.split())
    fig.savefig(os.path.join(figure_path, fig_name + ".png"))

# %% The contribution of main trading nations by BEC product category & trade typ
for flow in ("exports", "imports"):
    title_text1 = f"The contribution of top {flow[:-1]}ing EU Member States"
    title_text2 = "by BEC product category & trade type"
    fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(7, 10))
    for r, good in enumerate(('Capital goods', 'Consumption goods', 'Intermediate goods')):
        for c, trade in enumerate(('Extra-EU', 'Intra-EU')):
            tmp = share_prod_type_members[flow].loc[:, idx[trade, good]]
            tmp = tmp.nlargest(5)
            tmp.loc['Others'] = 100 - tmp.sum()
            tmp.plot.pie(ax=ax[r, c], autopct='%1.1f%%')
            ax[r, c].set_title(f"{trade} {good} {flow}")
            ax[r, c].set_ylabel(None)
    fig.suptitle(title_text1 + " \n " + title_text2)
    fig.tight_layout()
    fig_name = "_".join((f"Largest {flow[:-1]}ing Members States by BEC product category").split())
    fig.savefig(os.path.join(figure_path, fig_name + ".png"))

# %%
