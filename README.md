# Analysis of EU trade data

## Analysis of EU trade data for a typical period based on Eurostat data
The project includes python code needed to analyse the pattern of EU trade data for a typical period based on Eurostat data.

Python libraries used:
- Pandas
- Numpy
- Seaborn
- Matplotlib

## Installation
To install all of the libraries at once from a conda environment, run the following:

- `conda install numpy pandas matplotlib seaborn py7zr`

## Files
- **main.py:** Main python script file to produce intended tables and figures outputs.


## Information

### Visualisations
- **Pie Chart:** A pie chart a graph that illustrates the proportion represented by each components from the total quantity.

 ..* Example 1: EU's main trading partners and their contribution to Extra-EU trade
 ![alt-text](https://github.com/derecode/TradeDataAnalysis/blob/main/figures/Major_trading_partners_and_their_shares_in_Extra-EU.png)


..* Example 2: The contribution of capital, consumption and intermediate goods to Intra-Eu and Extra-EU trade

![alt-text](https://github.com/derecode/TradeDataAnalysis/blob/main/figures/EU_trade_flows_by_BEC_product_category_and_trade_type.png)

..* Additional figures that can be found under the `/figures` include:
...* The contribution of largest exporting Member States to EU trade: by product and trade type
  ![alt-text](https://github.com/derecode/TradeDataAnalysis/blob/main/figures/Largest_exporting_Members_States_by_BEC_product_category.png)
  
  
...* The contribution of largest importing Member States to EU trade: by product and trade type
 ![alt-text](https://github.com/derecode/TradeDataAnalysis/blob/main/figures/Largest_importing_Members_States_by_BEC_product_category.png)
  
- **Bar Plots:** Classical bar plots that are good for visualisation and comparison of different data statistics, especially comparing statistics of feature variables.

..* Example: The contribution of largest trading Member States to EU trade

 ![alt text](https://github.com/derecode/TradeDataAnalysis/blob/main/figures/Contribution_of_EU_Member_States_to_intra-EU_and_extra-EU_trade.png)
..* [Additional figures can be found under the `/figures` directory](https://github.com/derecode/TradeDataAnalysis/blob/main/figures), including 
...* 




- **Tables:** Example

|FLOW   |Extra-EU (trillion €)|Intra-EU (trillion €)|
|-------|---------------------|---------------------|
|exports|1.672                |2.888                |
|imports|1.573                |2.824                |

[Additional tables can be found under the `/tables` directory](https://github.com/derecode/TradeDataAnalysis/blob/main/tables)