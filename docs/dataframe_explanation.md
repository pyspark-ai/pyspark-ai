# DataFrame Explanation

## API
```python
DataFrame.ai.explain(cache: bool = True) -> str:
```

This method generates a natural language explanation of the SQL plan of the input Spark DataFrame.
## Example
Given a DataFrame `auto_df` from [Data Ingestion](data_ingestion.md), you can explain a DataFrame with the following code:
```python
auto_top_growth_df=auto_df.ai.transform("brand with the highest growth")
auto_top_growth_df.ai.explain()
```

> In summary, this dataframe is retrieving the brand with the highest sales change in 2022 compared to 2021. It presents the results sorted by sales change in descending order and only returns the top result.
