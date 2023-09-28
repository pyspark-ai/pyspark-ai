# Data Ingestion

## API
```python
SparkAI.create_df(
    desc: str,
    columns: Optional[List[str]] = None,
    cache: bool = True) -> DataFrame
```

This method create a Spark DataFrame by querying an LLM from web search result.

- param desc: the description of the result DataFrame, which will be used for web searching
- param columns: the expected column names in the result DataFrame
- param cache: If `True`, fetches cached data, if available. If `False`, retrieves fresh data and updates cache.
- return: a Spark DataFrame

## Example
If you have [set up the Google Python client](https://developers.google.com/docs/api/quickstart/python), you can ingest data via search engine:
```python
auto_df = spark_ai.create_df("2022 USA national auto sales by brand")
```
Otherwise, you can ingest data via URL:
```python
auto_df = spark_ai.create_df("https://www.carpro.com/blog/full-year-2022-national-auto-sales-by-brand")
```

Take a look at the data:
```python
auto_df.show(n=5)
```

| rank | brand     | us_sales_2022 | sales_change_vs_2021 |
|------|-----------|---------------|----------------------|
| 1    | Toyota    | 1849751       | -9                   |
| 2    | Ford      | 1767439       | -2                   |
| 3    | Chevrolet | 1502389       | 6                    |
| 4    | Honda     | 881201        | -33                  |
| 5    | Hyundai   | 724265        | -2                   |
