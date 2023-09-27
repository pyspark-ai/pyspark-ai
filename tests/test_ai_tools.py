import unittest
from unittest.mock import MagicMock
from langchain.base_language import BaseLanguageModel
from pyspark.sql import SparkSession

from pyspark_ai.pyspark_ai import SparkAI
from pyspark_ai.tool import QuerySparkSQLTool, QueryValidationTool, SimilarValueTool


class TestToolsInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.getOrCreate()
        cls.llm_mock = MagicMock(spec=BaseLanguageModel)

    def test_exclude_similar_value_tool(self):
        """Test that SimilarValueTool is excluded by default"""
        spark_ai = SparkAI(llm=self.llm_mock, spark_session=self.spark)
        agent = spark_ai._create_sql_agent()
        self.assertEqual(
            agent.tools,
            [
                QuerySparkSQLTool(spark=spark_ai._spark),
                QueryValidationTool(spark=spark_ai._spark),
            ],
        )

    def test_include_similar_value_tool(self):
        """Test that SimilarValueTool is included when vector_store_dir is specified"""
        vector_store_dir = "temp/"
        spark_ai = SparkAI(
            llm=self.llm_mock,
            spark_session=self.spark,
            vector_store_dir=vector_store_dir,
        )
        agent = spark_ai._create_sql_agent()
        self.assertEqual(
            agent.tools,
            [
                QuerySparkSQLTool(spark=spark_ai._spark),
                QueryValidationTool(spark=spark_ai._spark),
                SimilarValueTool(
                    spark=spark_ai._spark, vector_store_dir=vector_store_dir
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
