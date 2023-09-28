# DataFrame Transformation

## API
```python
DataFrame.ai.transform(desc: str, cache: bool = True) -> DataFrame
```

This method applies a transformation to a provided Spark DataFrame, the specifics of which are determined by the `desc` parameter:

- param desc: A natural language string that outlines the specific transformation to be applied on the DataFrame.
- param cache: If `True`, fetches cached data, if available. If `False`, retrieves fresh data and updates cache.
- return: Returns a new Spark DataFrame that is the result of applying the specified transformation
                 on the input DataFrame.

## Example
Given the following DataFrame `df`:
```python
df = spark_ai._spark.createDataFrame(
    [
        ("Normal", "Cellphone", 6000),
        ("Normal", "Tablet", 1500),
        ("Mini", "Tablet", 5500),
        ("Mini", "Cellphone", 5000),
        ("Foldable", "Cellphone", 6500),
        ("Foldable", "Tablet", 2500),
        ("Pro", "Cellphone", 3000),
        ("Pro", "Tablet", 4000),
        ("Pro Max", "Cellphone", 4500)
    ],
    ["product", "category", "revenue"]
)
```

You can write English to perform transformations. For example:
```python
df.ai.transform("What are the best-selling and the second best-selling products in every category?").show()
```

| product  |category| revenue |
|----------|--------|---------|
| Foldable |Cellphone| 6500    |
| Nromal   |Cellphone| 6000    |
| Mini      |Tablet| 5500    |
| Pro |Tablet| 4000    |


```python
df.ai.transform("Pivot the data by product and the revenue for each product").show()
```

| Category  | Normal | Mini | Foldable |  Pro | Pro Max |
|-----------|--------|------|----------|------|---------|
| Cellphone |   6000 | 5000 |     6500 | 3000 |    4500 |
| Tablet    |   1500 | 5500 |     2500 | 4000 |    null |

For a detailed walkthrough of the transformations, please refer to our [transform_dataframe.ipynb](https://github.com/databrickslabs/pyspark-ai/blob/master/examples/transform_dataframe.ipynb) notebook.