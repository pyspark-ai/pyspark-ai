import os
import shutil
import unittest
from unittest.mock import MagicMock, patch
from langchain.base_language import BaseLanguageModel
from pyspark.sql import SparkSession

from pyspark_ai.pyspark_ai import SparkAI
from pyspark_ai.tool import (
    QuerySparkSQLTool,
    QueryValidationTool,
    SimilarValueTool,
    VectorSearchUtil,
    LRUVectorStore,
)
from benchmark.wikisql.wiki_sql import (
    get_table_name,
    create_temp_view_statements,
    get_tables_and_questions,
)


class TestToolsInit(unittest.TestCase):
    """Tests initialization of SQL agent tools"""

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.getOrCreate()
        cls.llm_mock = MagicMock(spec=BaseLanguageModel)
        cls.vector_store_dir = "tests/data/vector_files/"

    @classmethod
    def tearDown(cls):
        """Remove vector files after each test"""
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
    def tearDown(cls):
        """Remove vector files after each test"""
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
                col_lst=dates,
                vector_store_path=None,
                lru_vector_store=None,
                search_text=search_date,
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

    @patch("pyspark_ai.tool.LRUVectorStore.get_file_size_bytes", return_value=1e9)
    def test_vector_file_lru_store_eviction(self, mock_get_file_size_bytes):
        """Tests LRUVectorStore LRU file eviction, using mocked get_file_size_bytes"""

        # mock the return value of LRUVectorStore.get_file_size_bytes, to always be 1 GB
        mock_get_file_size_bytes.return_value = 1e9

        spark_ai = SparkAI(
            llm=self.llm_mock,
            spark_session=self.spark,
            vector_store_dir=self.vector_store_dir,
            vector_store_max_gb=2,
        )
        agent = spark_ai._create_sql_agent()
        similar_value_tool = agent.lookup_tool("similar_value")

        table_file = "tests/data/test_transform_ai_tools.tables.jsonl"
        source_file = "tests/data/test_similar_value_tool_e2e.jsonl"

        stored_vector_files = []

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

                # test that stored vector files never exceed max size, 2GB
                self.assertTrue(len(os.listdir(self.vector_store_dir)) <= 2)

                # add only new file to stored_vector_files
                for file in os.listdir(self.vector_store_dir):
                    if file not in stored_vector_files:
                        stored_vector_files.append(file)
            finally:
                self.spark.sql(f"DROP TABLE IF EXISTS {table_name}")

        # check that only first file added was the one evicted
        self.assertTrue(len(os.listdir(self.vector_store_dir)) <= 2)
        self.assertTrue(
            sorted(os.listdir(self.vector_store_dir)) == sorted(stored_vector_files[1:])
        )

    def test_vector_file_lru_store_large_max_files(self):
        """Tests LRUVectorStore stores all vector files to disk with large max size, for 3 small dfs"""
        vector_store_max_gb = 100

        spark_ai = SparkAI(
            llm=self.llm_mock,
            spark_session=self.spark,
            vector_store_dir=self.vector_store_dir,
            vector_store_max_gb=vector_store_max_gb,
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
            finally:
                self.spark.sql(f"DROP TABLE IF EXISTS {table_name}")

        # test that all 3 vector files stored on disk
        self.assertTrue(len(os.listdir(self.vector_store_dir)) == 3)

    def test_vector_file_lru_store_zero_max_files(self):
        """Tests LRUVectorStore always evicts files when max dir size is 0"""
        vector_store_max_gb = 0

        spark_ai = SparkAI(
            llm=self.llm_mock,
            spark_session=self.spark,
            vector_store_dir=self.vector_store_dir,
            vector_store_max_gb=vector_store_max_gb,
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

                # test that number of vector files stored on disk never exceeds vector_store_max_gb, 0
                self.assertTrue(len(os.listdir(self.vector_store_dir)) == 0)
                self.assertTrue(LRUVectorStore.get_storage(self.vector_store_dir) == 0)
            finally:
                self.spark.sql(f"DROP TABLE IF EXISTS {table_name}")

    def test_vector_file_lru_store_prepopulated_dirs(self):
        """Tests LRUVectorStore counts vector stores already present in the vector store dir in eviction policy"""
        # add two dirs to vector_store_dir
        if not os.path.exists(self.vector_store_dir):
            os.makedirs(self.vector_store_dir)

        path1 = os.path.join(self.vector_store_dir, "dir1")
        path2 = os.path.join(self.vector_store_dir, "dir2")

        os.makedirs(path1)
        os.makedirs(path2)

        # check that vector_store_dir contains the 2 dirs
        self.assertTrue(len(os.listdir(self.vector_store_dir)) == 2)

        vector_store_max_gb = 0

        with self.assertRaises(Exception) as e:
            spark_ai = SparkAI(
                llm=self.llm_mock,
                spark_session=self.spark,
                vector_store_dir=self.vector_store_dir,
                vector_store_max_gb=vector_store_max_gb,
            )

            self.assertTrue("already exceeds max directory size" in e.exception)

    @unittest.skipUnless(
        os.environ.get("OPENAI_API_KEY") and os.environ["OPENAI_API_KEY"].strip() != "",
        "OPENAI_API_KEY is not set",
    )
    def test_transform_without_similar_value_tool(self):
        """Test that agent does not try to access SimilarValueTool if it is disabled"""
        spark_ai = SparkAI()

        # redirect verbose thoughts to f
        from contextlib import redirect_stdout
        import io

        f = io.StringIO()

        with redirect_stdout(f):
            table_file = "tests/data/test_transform_ai_tools.tables.jsonl"
            source_file = "tests/data/test_transform_without_similar_value_tool.jsonl"

            # prepare tables
            statements = create_temp_view_statements(table_file)
            for stmt in statements:
                self.spark.sql(stmt)

            (tables, questions, results, sqls) = get_tables_and_questions(source_file)

            for table, question, result, sql in zip(tables, questions, results, sqls):
                table_name = get_table_name(table)
                try:
                    df = self.spark.table(f"`{table_name}`")
                    df.createOrReplaceTempView(f"`{table_name}`")
                    spark_ai._get_transform_sql_query(df=df, desc=question, cache=False)
                finally:
                    self.spark.sql(f"DROP TABLE IF EXISTS {table_name}")

        # assert that 'similar_value' not attempted
        verbose_thoughts = f.getvalue()
        self.assertTrue("similar_value" not in verbose_thoughts)


if __name__ == "__main__":
    unittest.main()
