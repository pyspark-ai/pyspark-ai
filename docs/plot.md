# Plot

Given a DataFrame `auto_df` from [Data Ingestion](data_ingestion.md), you can plot it with the following code:
```python
auto_df.ai.plot()
```
![2022 USA national auto sales by brand](_static/auto_sales.png)

To plot with an instruction:
```python
auto_df.ai.plot("pie chart for US sales market shares, show the top 5 brands and the sum of others")
```
![2022 USA national auto sales_market_share by brand](_static/auto_sales_pie_char.png)