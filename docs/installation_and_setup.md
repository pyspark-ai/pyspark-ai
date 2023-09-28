# Installation and setup

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

