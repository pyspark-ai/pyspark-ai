# Introduction

The English SDK for Apache Spark is an extremely simple yet powerful tool. It takes English instructions and compile them into PySpark objects like DataFrames.
Its goal is to make Spark more user-friendly and accessible, allowing you to focus your efforts on extracting insights from your data.


![English As Code](_static/english_as_code.png)

## Getting Started
### DataFrame Transformation
Given the following DataFrame `df`, you can write English to transform it to another **DataFrame**. For example:
``` py
df.ai.transform("What are the best-selling and the second best-selling products in every category?").show()
```

| product  |category| revenue |
|----------|--------|---------|
| Foldable |Cellphone| 6500    |
| Nromal   |Cellphone| 6000    |
| Mini      |Tablet| 5500    |
| Pro |Tablet| 4000    |


### Data Ingestion
``` py
auto_df = spark_ai.create_df("2022 USA national auto sales by brand")
```

### Plot
``` py
auto_df.ai.plot("pie chart for US sales market shares, show the top 5 brands and the sum of others")
```
![2022 USA national auto sales_market_share by brand](_static/auto_sales_pie_char.png)

