# LLM Assistant for Apache Spark

## Installation

```bash
pip install spark-llm
```

## Usage
### Initialization
```python
from spark_llm import SparkLLMAssistant

assistant = SparkLLMAssistant()
assistant.activate() # active partial functions for Spark DataFrame
```

### Data Ingestion
```python
auto_df = assistant.create_df("2022 USA national auto sales by brand")
auto_df.show(n=5)
```
| rank | brand     | us_sales_2022 | sales_change_vs_2021 |
|------|-----------|---------------|----------------------|
| 1    | Toyota    | 1849751       | -9                   |
| 2    | Ford      | 1767439       | -2                   |
| 3    | Chevrolet | 1502389       | 6                    |
| 4    | Honda     | 881201        | -33                  |
| 5    | Hyundai   | 724265        | -2                   |

### Plot
```python
auto_df.llm.plot()
```
![2022 USA national auto sales by brand](docs/_static/auto_sales.png)

To plot with an instruction:
```python
auto_df.llm.plot("pie char for top 5 brands and the others' market shares")
```
![2022 USA national auto sales_market_share by brand](docs/_static/auto_sales_pie_char.png)
### DataFrame Transformation
```python
auto_top_growth_df=auto_df.llm.transform("top brand with the highest growth")
auto_top_growth_df.show()
```
| brand    | us_sales_2022 | sales_change_vs_2021 |
|----------|---------------|----------------------|
| Cadillac | 134726        | 14                   |

### DataFrame Explanation
```python
auto_top_growth_df.llm.explain()
```

> In summary, this dataframe is retrieving the brand with the highest sales change in 2022 compared to 2021. It presents the results sorted by sales change in descending order and only returns the top result.

### DataFrame Attribute Verification
```python
auto_top_growth_df.llm.verify("expect sales change percentage to be between -100 to 100")
```

> result: True

### UDF Generation
```python
@assistant.udf
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

### Cache
The SparkLLMAssistant supports a simple in-memory and persistent cache system. It keeps an in-memory staging cache that can be persisted through the `commit()` method. Cache lookup is always performed on the persistent cache only.

```python
assistant.commit()
```

Refer to [example.ipynb](https://github.com/gengliangwang/spark-llm/blob/main/examples/example.ipynb) for more detailed usage examples.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
Licensed under the Apache License 2.0.
