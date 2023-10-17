# Installation and Setup

## Installation

pyspark-ai can be installed via pip from [PyPI](https://pypi.org/project/pyspark-ai/):
```bash
pip install pyspark-ai
```

pyspark-ai can also be installed with optional dependencies to enable certain functionality. 
For example, to install pyspark-ai with the optional dependencies to ingest DataFrames:

```bash
pip install pyspark-ai[ingestion]
```

For a full list of optional dependencies, see the [Optional Dependencies](#optional-dependencies) section.

## Configuring OpenAI LLMs
As of July 2023, we have found that the GPT-4 works optimally with the English SDK. This superior AI model is readily accessible to all developers through the OpenAI API.

To use OpenAI's Language Learning Models (LLMs), you can set your OpenAI secret key as the `OPENAI_API_KEY` environment variable. This key can be found in your [OpenAI account](https://platform.openai.com/account/api-keys):
```bash
export OPENAI_API_KEY='sk-...'
```
By default, the `SparkAI` instances will use the GPT-4 model. However, you're encouraged to experiment with creating and implementing other LLMs, which can be passed during the initialization of `SparkAI` instances for various use-cases.

## Initialization

``` py
from pyspark_ai import SparkAI

spark_ai = SparkAI()
spark_ai.activate()  # active partial functions for Spark DataFrame
```

You can also pass other LLMs to construct the SparkAI instance. For example, by following [this guide](https://python.langchain.com/docs/integrations/chat/azure_chat_openai):
``` py
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

## Optional Dependencies

pyspark-ai has many optional dependencies that are only used for specific methods. 
For example, ingestion via `spark_ai.create_df("...")` requires the `requests` package, while plotting via `df.plot()` requires the `plotly` package. 
If the optional dependency is not installed, pyspark-ai will raise an Exception if a method requiring that dependency is called.

If using pip, optional pyspark-ai dependencies can be installed as optional extras, e.g. `pip install pyspark-ai[ingestion, plot]`. 
All optional dependencies can be installed with `pip install pyspark-ai[all]`.

Specific groups and their associated dependencies are listed below.

| Group       | Description | Dependencies | Pip extra
|-------------| ----------- |
| Plot        | Title       |
| Ingestion   | Text        |
| ----------- | ----------- |
| Header      | Title       |
| Paragraph   | Text        |
| ----------- | ----------- |
| Header      | Title       |
| Paragraph   | Text        |

[tool.poetry.group.plot.dependencies]
pandas = ">=1.0.5"
plotly = "^5.15.0"

[tool.poetry.group.vector-search.dependencies]
faiss-cpu = "^1.7.4"
sentence-transformers = "^2.2.2"
# avoid torch version poetry.lock bug https://github.com/pytorch/pytorch/issues/100974
torch = ">=2.0.0, !=2.0.1"

[tool.poetry.group.ingestion.dependencies]
requests = "^2.31.0"
tiktoken = "0.4.0"
beautifulsoup4 = "^4.12.2"
google-api-python-client = "^2.90.0"

[tool.poetry.group.spark-connect.dependencies]
grpcio = ">=1.56.0"
grpcio-status = ">=1.56.0"
pyarrow = ">=4.0.0"

[tool.poetry.group.dev.dependencies]
pyspark = "^3.4.0"
babel = "^2.12.1"
chispa = "^0.9.2"

[tool.poetry.group.lint.dependencies]
flake8 = "^6.0.0"
black = "^23.7.0"