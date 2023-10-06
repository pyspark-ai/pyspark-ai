import os
import shutil
import unittest
from unittest.mock import MagicMock
from langchain.base_language import BaseLanguageModel
from pyspark.sql import SparkSession

from pyspark_ai.pyspark_ai import SparkAI
from pyspark_ai.tool import (
    QuerySparkSQLTool,
    QueryValidationTool,
    SimilarValueTool,
    VectorSearchUtil,
)
from benchmark.wikisql.wiki_sql import (
    get_table_name,
    create_temp_view_statements,
)


class TestToolsInit(unittest.TestCase):
    """Tests initialization of SQL agent tools"""

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.getOrCreate()
        cls.llm_mock = MagicMock(spec=BaseLanguageModel)
        cls.vector_store_dir = "tests/data/vector_files/"

    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree("tests/data/vector_files/")
        except Exception:
            pass

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
        spark_ai = SparkAI(
            llm=self.llm_mock,
            spark_session=self.spark,
            vector_store_dir=self.vector_store_dir,
            vector_cache_max_files=64
        )
        agent = spark_ai._create_sql_agent()
        tools = agent.tools

        self.assertTrue(isinstance(tools[0], QuerySparkSQLTool))
        self.assertTrue(isinstance(tools[1], QueryValidationTool))
        self.assertTrue(isinstance(tools[2], SimilarValueTool))


class TestSimilarValueTool(unittest.TestCase):
    """Tests SimilarValueTool functionality"""

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.getOrCreate()
        cls.llm_mock = MagicMock(spec=BaseLanguageModel)
        cls.vector_store_dir = "tests/data/vector_files/"

    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree("tests/data/vector_files/")
        except Exception:
            pass

    def test_similar_dates(self):
        """Test retrieval of similar date by similarity_vector_search"""
        dates = [
            "2023-04-15",
            "2022-09-28",
            "2011-01-10",
            "2022-11-05",
            "2023-08-20",
            "2019-07-12",
            "2023-03-25",
            "2002-05-30",
            "2023-06-08",
            "20007-12-03",
        ]

        # try different formats of date "2023-03-25"
        search_dates = [
            "March 25, 2023",
            "March 25th",
            "03/25/2023",
            "03/25/23",
            "3/25",
        ]

        for search_date in search_dates:
            similar_value = VectorSearchUtil.vector_similarity_search(
                col_lst=dates, vector_store_path=None, vector_file_cache=None, search_text=search_date
            )
            self.assertEqual(similar_value, "2023-03-25")

    @staticmethod
    def get_expected_results(inputs_file):
        """Helper util to get inputs for testing SimilarValueTool"""
        import json

        tables = []
        inputs = []
        results = []

        with open(inputs_file, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                tables.append(item["table_id"])
                inputs.append(item["tool_input"])
                results.append(item["result"])

        return tables, inputs, results

    def test_similar_value_tool_e2e(self):
        """End-to-end tests of similar value tool, using WikiSQL training tables"""
        spark_ai = SparkAI(
            llm=self.llm_mock,
            spark_session=self.spark,
            vector_store_dir=self.vector_store_dir,
        )
        agent = spark_ai._create_sql_agent()
        similar_value_tool = agent.lookup_tool("similar_value")

        table_file = "tests/data/test_transform_ai_tools.tables.jsonl"
        source_file = "tests/data/test_similar_value_tool_e2e.jsonl"

        # prepare tables
        statements = create_temp_view_statements(table_file)
        for stmt in statements:
            self.spark.sql(stmt)

        (
            tables,
            tool_inputs,
            expected_results,
        ) = TestSimilarValueTool.get_expected_results(source_file)

        for table, tool_input, expected_result in zip(
            tables, tool_inputs, expected_results
        ):
            table_name = get_table_name(table)
            try:
                df = self.spark.table(f"`{table_name}`")
                df.createOrReplaceTempView(f"`{table_name}`")
                observation = similar_value_tool.run(f"{tool_input}{table_name}")

                self.assertEqual(observation, expected_result)
            finally:
                self.spark.sql(f"DROP TABLE IF EXISTS {table_name}")

    def test_vector_file_lru_cache_max_files(self):
        """Tests VectorFileLRUCache adheres to max file size, using WikiSQL training tables"""
        # set max vector files to 2
        spark_ai = SparkAI(
            llm=self.llm_mock,
            spark_session=self.spark,
            vector_store_dir=self.vector_store_dir,
            vector_cache_max_files=2
        )
        agent = spark_ai._create_sql_agent()
        similar_value_tool = agent.lookup_tool("similar_value")

        table_file = "tests/data/test_transform_ai_tools.tables.jsonl"
        source_file = "tests/data/test_similar_value_tool_e2e.jsonl"

        # prepare tables
        statements = create_temp_view_statements(table_file)
        for stmt in statements:
            self.spark.sql(stmt)

        (
            tables,
            tool_inputs,
            expected_results,
        ) = TestSimilarValueTool.get_expected_results(source_file)

        for table, tool_input, expected_result in zip(
            tables, tool_inputs, expected_results
        ):
            table_name = get_table_name(table)
            try:
                df = self.spark.table(f"`{table_name}`")
                df.createOrReplaceTempView(f"`{table_name}`")
                similar_value_tool.run(f"{tool_input}{table_name}")

                print("*****len", len(os.listdir(self.vector_store_dir)))
                print("*****list", os.listdir(self.vector_store_dir))

                # test that number of vector files stored on disk never exceeds 2
                self.assertTrue(len(os.listdir(self.vector_store_dir)) <= 2)
            finally:
                self.spark.sql(f"DROP TABLE IF EXISTS {table_name}")


if __name__ == "__main__":
    unittest.main()
