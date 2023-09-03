<div align="center">

![English SDK for Apache Spark](./docs/_static/english-sdk-spark.svg)
</div>

## Introduction
The English SDK for Apache Spark is an extremely simple yet powerful tool. It takes English instructions and compile them into PySpark objects like DataFrames.
Its goal is to make Spark more user-friendly and accessible, allowing you to focus your efforts on extracting insights from your data.

For a more comprehensive introduction and background to our project, we have the following resources:
- [Blog Post](https://www.databricks.com/blog/introducing-english-new-programming-language-apache-spark): A detailed walkthrough of our project.
- [Demo Video](https://www.youtube.com/watch?v=yj7XlTB1Jvc&t=511s): 2023 Data + AI summit announcement video with demo.
- [Breakout Session](https://www.youtube.com/watch?v=ZunjkL3L62o&t=73s): A deep dive into the story behind the English SDK, its features, and future works at DATA+AI summit 2023.

## Installation

```bash
pip install pyspark-ai
```

## Configuring OpenAI LLMs
As of July 2023, we have found that the GPT-4 works optimally with the English SDK. This superior AI model is readily accessible to all developers through the OpenAI API.

To use OpenAI's Language Learning Models (LLMs), you can set your OpenAI secret key as the `OPENAI_API_KEY` environment variable. This key can be found in your [OpenAI account](https://platform.openai.com/account/api-keys). Example:
```bash
export OPENAI_API_KEY='sk-...'
```
By default, the `SparkAI` instances will use the GPT-4 model. However, you're encouraged to experiment with creating and implementing other LLMs, which can be passed during the initialization of `SparkAI` instances for various use-cases.

## Usage
### Initialization

```python
from pyspark_ai import SparkAI

spark_ai = SparkAI()
spark_ai.activate()  # active partial functions for Spark DataFrame
```

You can also pass other LLMs to construct the SparkAI instance. For example, by following [this guide](https://python.langchain.com/docs/integrations/chat/azure_chat_openai):
```python
from langchain.chat_models import AzureChatOpenAI
from pyspark_ai import SparkAI

llm = AzureChatOpenAI(
    deployment_name=...,
    model_name=...
)
spark_ai = SparkAI(llm=llm)
spark_ai.activate()  # active partial functions for Spark DataFrame
```
As per [Microsoft's Data Privacy page](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/data-privacy), using the Azure OpenAI service can provide better data privacy and security.

### DataFrame Transformation
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

We're delighted that you're considering contributing to the English SDK for Apache Spark project! Whether you're fixing a bug or proposing a new feature, your contribution is highly appreciated.

Before you start, please take a moment to read our [Contribution Guide](./CONTRIBUTING.md). This guide provides an overview of how you can contribute to our project. We're currently in the early stages of development and we're working on introducing more comprehensive test cases and Github Action jobs for enhanced testing of each pull request.

If you have any questions or need assistance, feel free to open a new issue in the GitHub repository.

Thank you for helping us improve the English SDK for Apache Spark. We're excited to see your contributions!

## License
Licensed under the Apache License 2.0.
