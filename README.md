<div align="center">

![English SDK for Apache Spark](./docs/_static/english-sdk-spark.svg)
<h2>
English is the new programming language;<br />
Generative AI is the new compiler;<br />
Python is ... the new byte code.
</h2>
</div>

## Installation

```bash
pip install pyspark-ai
```

## Configuring OpenAI LLMs
As of June 2023, our extensive testing suggests the most effective utilization with the English SDK and GPT-4.

To use OpenAI's Language Learning Models (LLMs), you can set your OpenAI secret key as the `OPENAI_API_KEY` environment variable. This key can be found in your [OpenAI account](https://platform.openai.com/account/api-keys). Example:
```bash
export OPENAI_API_KEY='sk-...'
```
By default, the `SparkAI` instances will use the GPT-4 model. However, you're encouraged to experiment with creating and implementing other LLMs, which can be passed during the initialization of `SparkAI` instances for various use-cases.

## Usage
### Initialization

```python
from langchain.chat_models import ChatOpenAI
from pyspark_ai import SparkAI

# Initialize ChatOpenAI with 'gpt-4'
# If 'gpt-4' is unavailable, use 'gpt-3.5-turbo' (might lower output quality)
llm = ChatOpenAI(model_name='gpt-4', temperature=0)

# Initialize SparkAI with the ChatOpenAI model
spark_ai = SparkAI(llm=llm, verbose=True)

# Activate partial functions for Spark DataFrame
spark_ai.activate()
```

### Data Ingestion
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

### Plot
```python
auto_df.ai.plot()
```
![2022 USA national auto sales by brand](docs/_static/auto_sales.png)

To plot with an instruction:
```python
auto_df.ai.plot("pie chart for US sales market shares, show the top 5 brands and the sum of others")
```
![2022 USA national auto sales_market_share by brand](docs/_static/auto_sales_pie_char.png)
### DataFrame Transformation
```python
auto_top_growth_df=auto_df.ai.transform("brand with the highest growth")
auto_top_growth_df.show()
```
| brand    | us_sales_2022 | sales_change_vs_2021 |
|----------|---------------|----------------------|
| Cadillac | 134726        | 14                   |

### DataFrame Explanation
```python
auto_top_growth_df.ai.explain()
```

> In summary, this dataframe is retrieving the brand with the highest sales change in 2022 compared to 2021. It presents the results sorted by sales change in descending order and only returns the top result.

### DataFrame Attribute Verification
```python
auto_top_growth_df.ai.verify("expect sales change percentage to be between -100 to 100")
```

> result: True

### UDF Generation
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

### Cache
The SparkAI supports a simple in-memory and persistent cache system. It keeps an in-memory staging cache, which gets updated for LLM and web search results. The staging cache can be persisted through the commit() method. Cache lookup is always performed on both in-memory staging cache and persistent cache.
```python
spark_ai.commit()
```

Refer to [example.ipynb](https://github.com/databrickslabs/pyspark-ai/blob/master/examples/example.ipynb) for more detailed usage examples.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
Licensed under the Apache License 2.0.
