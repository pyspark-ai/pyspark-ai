# LLM Assistant for Apache Spark

## Installation

```bash
pip install spark-llm
```

## Usage
### Initialization
```python
from langchain.chat_models import ChatOpenAI
from spark_llm import SparkLLMAssistant

llm = ChatOpenAI(model_name='gpt-4') # using gpt-4 can achieve better results
assistant=SparkLLMAssistant(llm=llm)
assistant.activate() # active partial functions for Spark DataFrame
```

### Data Ingestion
```python
auto_df=assistant.create_df("2022 USA national auto sales by brand")
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
auto_df.llm_plot()
```
![2022 USA national auto sales by brand](docs/_static/auto_sales.png)
### DataFrame Transformation
```python
auto_top_growth_df=auto_df.llm_transform("top brand with the highest growth")
auto_top_growth_df.show()
```
| brand    | us_sales_2022 | sales_change_vs_2021 |
|----------|---------------|----------------------|
| Cadillac | 134726        | 14                   |

### DataFrame Explanation
```python
auto_top_growth_df.llm_explain()
```

> In summary, this dataframe is retrieving the brand with the highest sales change in 2022 compared to 2021. It presents the results sorted by sales change in descending order and only returns the top result.

Refer to [example.ipynb](https://github.com/gengliangwang/spark-llm/blob/main/examples/example.ipynb) for more detailed usage examples.

### Test Generation
```python
import pyspark.sql.functions as F

def remove_non_word_characters(col):
    return F.regexp_replace(col, "[^\\w\\s]+", "")

# ask the assistant to generate tests for the df transform function
assistant.test_llm(remove_non_word_characters)
```

```python
import unittest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StringType

def remove_non_word_characters(df, input_col):
    return df.withColumn(input_col, col(input_col).regexp_replace('[^a-zA-Z0-9]+', ''))

class TestRemoveNonWordCharacters(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local[*]").appName("RemoveNonWordCharactersTest").getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_remove_non_word_characters(self):
        input_data = [("Test!@#",), ("123$%^",), ("No-Special",)]
        expected_output_data = [("Test",), ("123",), ("NoSpecial",)]

        input_schema = ["text"]
        expected_output_schema = ["text"]

        input_df = self.spark.createDataFrame(input_data, input_schema)
        expected_output_df = self.spark.createDataFrame(expected_output_data, expected_output_schema)

        output_df = remove_non_word_characters(input_df, "text")

        self.assertTrue(output_df.subtract(expected_output_df).count() == 0)
        self.assertTrue(expected_output_df.subtract(output_df).count() == 0)

    def test_remove_non_word_characters_empty_string(self):
        input_data = [("",)]
        expected_output_data = [("",)]

        input_schema = ["text"]
        expected_output_schema = ["text"]

        input_df = self.spark.createDataFrame(input_data, input_schema)
        expected_output_df = self.spark.createDataFrame(expected_output_data, expected_output_schema)

        output_df = remove_non_word_characters(input_df, "text")

        self.assertTrue(output_df.subtract(expected_output_df).count() == 0)
        self.assertTrue(expected_output_df.subtract(output_df).count() == 0)

result = unittest.main(argv=['first-arg-is-ignored'], exit=False)
```
> result: True

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
Licensed under the Apache License 2.0.
