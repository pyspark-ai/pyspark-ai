# UDF Generation

## Example 1: Compute expression from columns

You can use the `@spark_ai.udf` decorator to generate UDFs from Python functions. There is no need to implement the body of the method.

For example, given a DataFrame `auto_df` from [Data Ingestion](data_ingestion.md):
```python
@spark_ai.udf
def previous_years_sales(brand: str, current_year_sale: int, sales_change_percentage: float) -> int:
    """Calculate previous years sales from sales change percentage"""
    ...
    
spark.udf.register("previous_years_sales", previous_years_sales)
auto_df.createOrReplaceTempView("autoDF")

spark.sql("select brand as brand, previous_years_sales(brand, us_sales, sales_change_percentage) as 2021_sales from autoDF").show()
```

| brand         |2021_sales|
|---------------|-------------|
| Toyota        |   2032693|
| Ford          |   1803509|
| Chevrolet     |   1417348|
| Honda         |   1315225|
| Hyundai       |    739045|

## Example 2: Parse heterogeneous JSON text

```python
from typing import List

@spark_ai.udf
def parse_heterogeneous_json(json_str: str, schema: List[str]) -> List[str]:
    """Extract fields from heterogeneous JSON string based on given schema in a right order.
    If field is missing replace it by None. All imports should be inside function."""
    ...
```

`df.show()`

|          json_field|
|--------------------|
|{"address": "123 ...|
|{"last_name": "Do...|
|{"email": "john_d...|
|{"email": "john_d...|
|{"phone_number": ...|
|{"age": 32, "firs...|
|{"last_name": "Do...|
|{"last_name": "Do...|

```python
(
    bad_json_dataframe
    .withColumn(
        "parsed",
        expr("parse_heterogeneous_json(json_field, schema)"),
    )
    .select("parsed")
    .show()
```

|              parsed|
|--------------------|
|[NULL, John, Doe,...|
|[NULL, NULL, Doe,...|
|[NULL, John, NULL...|
|[NULL, NULL, Doe,...|
|[1279, NULL, NULL...|
|[1279, John, Doe,...|
|[1279, NULL, Doe,...|
|[NULL, John, Doe,...|


## Example 3: Extract email from raw text

```python
df = spark.createDataFrame(
    [
        "For any queries regarding the product, contact helpdesk@example.com.",
        "Send your applications to hr@ourcompany.com.",
        "You can reach out to the teacher at prof.mike@example.edu.",
        "My personal email is jane.doe@example.com.",
        "You can forward the documents to admin@oursite.net.",
    ],
    schema="string",
)
```

```python
@spark_ai.udf
def extract_email(text: str) -> str:
    """Extract first email from raw text"""
    ...

df.select(expr("extract_email(value)")).show()
```

|extract_email(value)|
|--------------------|
|helpdesk@example.com|
|   hr@ourcompany.com|
|prof.mike@example...|
|jane.doe@example.com|
|   admin@oursite.net|

## Example 4: Generate random numbers from Laplace distribution

```python
@spark_ai.udf
def laplace_random_number(loc: float, scale: float) -> float:
    """Generate a random number from Laplace distribution with given loc and scale in pure Python. Function should contain all necessary imports."""
    ...
```

```python
from pyspark.sql.functions import lit
from pyspark.sql.types import DoubleType

spark.udf.register("laplace_random_number", laplace_random_number, returnType=DoubleType())
(
    spark.sparkContext.range(0, 500_000)
    .toDF(schema="int")
    .withColumn("loc", lit(1.0).cast("double"))
    .withColumn("scale", lit(0.3).cast("double"))
    .withColumn("laplace_random", expr("laplace_random_number(loc, scale)"))
    .select("laplace_random")
    .show()
)
```

|     laplace_random|
|-------------------|
|0.39071216071827797|
| 0.4670818035437042|
| 0.7586462538760413|
|0.41987361759910846|
| 1.1906543111637395|
| 0.5811271918788534|
| 0.8442334249218986|
