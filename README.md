<div align="center">

![English SDK for Apache Spark](./docs/_static/english-sdk-spark.svg)
</div>

![![image](https://github.com/pyspark-ai/pyspark-ai/actions/workflows/build_and_test.yml/badge.svg?branch=master)](https://github.com/pyspark-ai/pyspark-ai/actions/workflows/build_and_test.yml/badge.svg?branch=master)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pyspark-ai)
[![PyPI version](https://badge.fury.io/py/pyspark-ai.svg)](https://badge.fury.io/py/pyspark-ai)

## Introduction
The English SDK for Apache Spark is an extremely simple yet powerful tool. It takes English instructions and compile them into PySpark objects like DataFrames.
Its goal is to make Spark more user-friendly and accessible, allowing you to focus your efforts on extracting insights from your data.

For a more comprehensive introduction and background to our project, we have the following resources:
- [Blog Post](https://www.databricks.com/blog/introducing-english-new-programming-language-apache-spark): A detailed walkthrough of our project.
- [Demo Video](https://www.youtube.com/watch?v=yj7XlTB1Jvc&t=511s): 2023 Data + AI summit announcement video with demo.
- [Breakout Session](https://www.youtube.com/watch?v=ZunjkL3L62o&t=73s): A deep dive into the story behind the English SDK, its features, and future works at DATA+AI summit 2023.

## Installation

pyspark-ai can be installed via pip from [PyPI](https://pypi.org/project/pyspark-ai/):
```bash
pip install pyspark-ai
```

pyspark-ai can also be installed with optional dependencies to enable certain functionality. 
For example, to install pyspark-ai with the optional dependencies to plot data from a DataFrame:

```bash
pip install "pyspark-ai[plot]"
```

To install all optionall dependencies:

```bash
pip install "pyspark-ai[all]"
```

For a full list of optional dependencies, see [Installation and Setup](./docs/installation_and_setup.md).

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
Using the Azure OpenAI service can provide better data privacy and security, as per [Microsoft's Data Privacy page](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/data-privacy).

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

### Transform Accuracy Improvement: Vector Similarity Search 

To improve the accuracy of transform query generation, you can also optionally enable vector similarity search. 
This is done by specifying a `vector_store_dir` location for the vector files when you initialize SparkAI. For example:

```python
from pyspark_ai import SparkAI

spark_ai = SparkAI(vector_store_dir="vector_store/") # vector files will be stored in the dir "vector_store"
spark_ai.activate() 
```

Now when you call df.ai.transform as before, the agent will use word embeddings to generate accurate query values.

For a detailed walkthrough, please refer to our [vector_similarity_search.ipynb](./examples/vector_similarity_search.ipynb).

### Plot
Let's create a DataFrame for car sales in the U.S.

```python
# auto sales data from https://www.carpro.com/blog/full-year-2022-national-auto-sales-by-brand
data = [('Toyota', 1849751, -9), ('Ford', 1767439, -2), ('Chevrolet', 1502389, 6),
        ('Honda', 881201, -33), ('Hyundai', 724265, -2), ('Kia', 693549, -1),
        ('Jeep', 684612, -12), ('Nissan', 682731, -25), ('Subaru', 556581, -5),
        ('Ram Trucks', 545194, -16), ('GMC', 517649, 7), ('Mercedes-Benz', 350949, 7),
        ('BMW', 332388, -1), ('Volkswagen', 301069, -20), ('Mazda', 294908, -11),
        ('Lexus', 258704, -15), ('Dodge', 190793, -12), ('Audi', 186875, -5),
        ('Cadillac', 134726, 14), ('Chrysler', 112713, -2), ('Buick', 103519, -42),
        ('Acura', 102306, -35), ('Volvo', 102038, -16), ('Mitsubishi', 102037, -16),
        ('Lincoln', 83486, -4), ('Porsche', 70065, 0), ('Genesis', 56410, 14),
        ('INFINITI', 46619, -20), ('MINI', 29504, -1), ('Alfa Romeo', 12845, -30),
        ('Maserati', 6413, -10), ('Bentley', 3975, 0), ('Lamborghini', 3134, 3),
        ('Fiat', 915, -61), ('McLaren', 840, -35), ('Rolls-Royce', 460, 7)]

auto_df = spark_ai._spark.createDataFrame(data, ["Brand", "US_Sales_2022", "Sales_Change_Percentage"])
```

We can visualize the data with the plot API:

```python
# call plot() with no args for LLM-generated plot
auto_df.ai.plot()
```
![2022 USA national auto sales by brand](docs/_static/auto_sales.png)

To plot with an instruction:
```python
auto_df.ai.plot("pie chart for US sales market shares, show the top 5 brands and the sum of others")
```
![2022 USA national auto sales_market_share by brand](docs/_static/auto_sales_pie_char.png)

Please refer to [example.ipynb](https://github.com/databrickslabs/pyspark-ai/blob/master/examples/example.ipynb) for more APIs and detailed usage examples.

## Contributing

We're delighted that you're considering contributing to the English SDK for Apache Spark project! Whether you're fixing a bug or proposing a new feature, your contribution is highly appreciated.

Before you start, please take a moment to read our [Contribution Guide](./CONTRIBUTING.md). This guide provides an overview of how you can contribute to our project. We're currently in the early stages of development and we're working on introducing more comprehensive test cases and Github Action jobs for enhanced testing of each pull request.

If you have any questions or need assistance, feel free to open a new issue in the GitHub repository.

Thank you for helping us improve the English SDK for Apache Spark. We're excited to see your contributions!

## License
Licensed under the Apache License 2.0.
