# UDF Generation
## API
```python
@spark_ai.udf
def udf_name(arg1: arg1_type, arg2: arg2_type, ...) -> return_type:
    """UDF description"""
    ...
```
Given a SparkAI instance `spark_ai`, you can use the `@spark_ai.udf` decorator to generate UDFs from Python functions. There is no need to implement the body of the method.

## Example 1: Compute expression from columns
Given a DataFrame `auto_df` from [Data Ingestion](data_ingestion.md):
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

Let's imagine we have heterogeneous JSON texts: each of them may contain or not keys and also order of keys is random. We can generate sucha a DataFrame by mutating single JSON.

```python
random_dict = {
    "id": 1279,
    "first_name": "John",
    "last_name": "Doe",
    "username": "johndoe",
    "email": "john_doe@example.com",
    "phone_number": "+1 234 567 8900",
    "address": "123 Main St, Springfield, OH, 45503, USA",
    "age": 32,
    "registration_date": "2020-01-20T12:12:12Z",
    "last_login": "2022-03-21T07:25:34Z",
}
original_keys = list(random_dict.keys())

from random import random, shuffle

mutaded_rows = []
for _ in range(20):
    keys = [k for k in original_keys]
    shuffle(keys)
    # With 0.4 chance drop each field and also shuffle an order
    mutaded_rows.append({k: random_dict[k] for k in keys if random() <= 0.6})

import json

bad_json_dataframe = (
    spark.createDataFrame(
        [(json.dumps(val), original_keys) for val in mutaded_rows],
        ["json_field", "schema"],
    )
)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>json_field</th>
      <th>schema</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>{"first_name": "John", "email": "john_doe@example.com", "last_name": "Doe", "phone_number": "+1 234 567 8900", "age": 32, "last_login": "2022-03-21T07:25:34Z", "address": "123 Main St, Springfield, OH, 45503, USA"}</td>
      <td>[id, first_name, last_name, username, email, phone_number, address, age, registration_date, last_login]</td>
    </tr>
    <tr>
      <td>{"address": "123 Main St, Springfield, OH, 45503, USA", "phone_number": "+1 234 567 8900", "email": "john_doe@example.com", "registration_date": "2020-01-20T12:12:12Z", "username": "johndoe", "last_login": "2022-03-21T07:25:34Z"}</td>
      <td>[id, first_name, last_name, username, email, phone_number, address, age, registration_date, last_login]</td>
    </tr>
    <tr>
      <td>{"age": 32, "last_name": "Doe", "email": "john_doe@example.com", "last_login": "2022-03-21T07:25:34Z", "address": "123 Main St, Springfield, OH, 45503, USA", "username": "johndoe"}</td>
      <td>[id, first_name, last_name, username, email, phone_number, address, age, registration_date, last_login]</td>
    </tr>
    <tr>
      <td>{"first_name": "John", "address": "123 Main St, Springfield, OH, 45503, USA", "phone_number": "+1 234 567 8900", "last_name": "Doe", "id": 1279}</td>
      <td>[id, first_name, last_name, username, email, phone_number, address, age, registration_date, last_login]</td>
    </tr>
    <tr>
      <td>{"phone_number": "+1 234 567 8900", "registration_date": "2020-01-20T12:12:12Z", "email": "john_doe@example.com", "address": "123 Main St, Springfield, OH, 45503, USA", "age": 32, "username": "johndoe"}</td>
      <td>[id, first_name, last_name, username, email, phone_number, address, age, registration_date, last_login]</td>
    </tr>
  </tbody>
</table>

```python
from typing import List

@spark_ai.udf
def parse_heterogeneous_json(json_str: str, schema: List[str]) -> List[str]:
    """Extract fields from heterogeneous JSON string based on given schema in a right order.
    If field is missing replace it by None. All imports should be inside function."""
    ...
```

Now we can test processing of our text rows:

```python
from pyspark.sql.functions import expr

(
    bad_json_dataframe
    .withColumn(
        "parsed",
        expr("parse_heterogeneous_json(json_field, schema)"),
    )
    .select("parsed")
    .show()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>json_field</th>
      <th>schema</th>
      <th>parsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>{"first_name": "John", "email": "john_doe@example.com", "last_name": "Doe", "phone_number": "+1 234 567 8900", "age": 32, "last_login": "2022-03-21T07:25:34Z", "address": "123 Main St, Springfield, OH, 45503, USA"}</td>
      <td>[id, first_name, last_name, username, email, phone_number, address, age, registration_date, last_login]</td>
      <td>[None, John, Doe, None, john_doe@example.com, +1 234 567 8900, 123 Main St, Springfield, OH, 45503, USA, 32, None, 2022-03-21T07:25:34Z]</td>
    </tr>
    <tr>
      <td>{"address": "123 Main St, Springfield, OH, 45503, USA", "phone_number": "+1 234 567 8900", "email": "john_doe@example.com", "registration_date": "2020-01-20T12:12:12Z", "username": "johndoe", "last_login": "2022-03-21T07:25:34Z"}</td>
      <td>[id, first_name, last_name, username, email, phone_number, address, age, registration_date, last_login]</td>
      <td>[None, None, None, johndoe, john_doe@example.com, +1 234 567 8900, 123 Main St, Springfield, OH, 45503, USA, None, 2020-01-20T12:12:12Z, 2022-03-21T07:25:34Z]</td>
    </tr>
    <tr>
      <td>{"age": 32, "last_name": "Doe", "email": "john_doe@example.com", "last_login": "2022-03-21T07:25:34Z", "address": "123 Main St, Springfield, OH, 45503, USA", "username": "johndoe"}</td>
      <td>[id, first_name, last_name, username, email, phone_number, address, age, registration_date, last_login]</td>
      <td>[None, None, Doe, johndoe, john_doe@example.com, None, 123 Main St, Springfield, OH, 45503, USA, 32, None, 2022-03-21T07:25:34Z]</td>
    </tr>
    <tr>
      <td>{"first_name": "John", "address": "123 Main St, Springfield, OH, 45503, USA", "phone_number": "+1 234 567 8900", "last_name": "Doe", "id": 1279}</td>
      <td>[id, first_name, last_name, username, email, phone_number, address, age, registration_date, last_login]</td>
      <td>[1279, John, Doe, None, None, +1 234 567 8900, 123 Main St, Springfield, OH, 45503, USA, None, None, None]</td>
    </tr>
    <tr>
      <td>{"phone_number": "+1 234 567 8900", "registration_date": "2020-01-20T12:12:12Z", "email": "john_doe@example.com", "address": "123 Main St, Springfield, OH, 45503, USA", "age": 32, "username": "johndoe"}</td>
      <td>[id, first_name, last_name, username, email, phone_number, address, age, registration_date, last_login]</td>
      <td>[None, None, None, johndoe, john_doe@example.com, +1 234 567 8900, 123 Main St, Springfield, OH, 45503, USA, 32, 2020-01-20T12:12:12Z, None]</td>
    </tr>
  </tbody>
</table>

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

from pyspark.sql.functions import col

spark.udf.register("extract_email", extract_email)
df.select(col("value").alias("raw"), expr("extract_email(value)").alias("email")).show()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>raw</th>
      <th>email</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>For any queries regarding the product, contact helpdesk@example.com.</td>
      <td>helpdesk@example.com</td>
    </tr>
    <tr>
      <td>Send your applications to hr@ourcompany.com.</td>
      <td>hr@ourcompany.com</td>
    </tr>
    <tr>
      <td>You can reach out to the teacher at prof.mike@example.edu.</td>
      <td>prof.mike@example.edu</td>
    </tr>
    <tr>
      <td>My personal email is jane.doe@example.com.</td>
      <td>jane.doe@example.com</td>
    </tr>
    <tr>
      <td>You can forward the documents to admin@oursite.net.</td>
      <td>admin@oursite.net</td>
    </tr>
  </tbody>
</table>

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

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>value</th>
      <th>loc</th>
      <th>scale</th>
      <th>laplace_random</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.0</td>
      <td>0.3</td>
      <td>0.799962</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.0</td>
      <td>0.3</td>
      <td>0.995381</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.0</td>
      <td>0.3</td>
      <td>0.602727</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.0</td>
      <td>0.3</td>
      <td>1.235575</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.0</td>
      <td>0.3</td>
      <td>1.864565</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.0</td>
      <td>0.3</td>
      <td>1.220493</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1.0</td>
      <td>0.3</td>
      <td>0.992431</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1.0</td>
      <td>0.3</td>
      <td>1.630307</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1.0</td>
      <td>0.3</td>
      <td>0.894683</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1.0</td>
      <td>0.3</td>
      <td>0.632602</td>
    </tr>
  </tbody>
</table>
