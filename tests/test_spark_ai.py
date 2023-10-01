import logging
import unittest
from io import StringIO
from unittest.mock import MagicMock

import numpy as np
from chispa.dataframe_comparer import assert_df_equality
from langchain.base_language import BaseLanguageModel
from pyspark.sql import Row, SparkSession
from pyspark_ai import SparkAI
from pyspark_ai.search_tool_with_cache import SearchToolWithCache
from tiktoken import Encoding


class SparkAIInitializationTestCase(unittest.TestCase):
    """Test cases for SparkAI initialization."""

    def setUp(self):
        self.llm_mock = MagicMock(spec=BaseLanguageModel)
        self.spark_session_mock = MagicMock(spec=SparkSession)
        self.encoding_mock = MagicMock(spec=Encoding)
        self.web_search_tool_mock = MagicMock(spec=SearchToolWithCache.search)
        self.spark_ai = SparkAI(
            llm=self.llm_mock,
            web_search_tool=self.web_search_tool_mock,
            spark_session=self.spark_session_mock,
            enable_cache=False,
            encoding=self.encoding_mock,
        )

    def test_initialization_with_default_values(self):
        """Tests if the class initializes correctly with default values."""
        self.assertEqual(self.spark_ai._spark, self.spark_session_mock)
        self.assertEqual(self.spark_ai._llm, self.llm_mock)
        self.assertEqual(self.spark_ai._web_search_tool, self.web_search_tool_mock)
        self.assertEqual(self.spark_ai._encoding, self.encoding_mock)
        self.assertEqual(self.spark_ai._max_tokens_of_web_content, 3000)


class TestGetTableCommentFromExplain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.getOrCreate()
        cls.llm_mock = MagicMock(spec=BaseLanguageModel)
        cls.spark_ai = SparkAI(llm=cls.llm_mock, spark_session=cls.spark)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def create_and_read_table(self, table_name, data, comment=""):
        self.spark.createDataFrame(data, ["col1", "col2"]).write.saveAsTable(table_name)
        if comment != "":
            self.spark.sql(
                f"ALTER TABLE {table_name} SET TBLPROPERTIES ('comment' = '{comment}')"
            )
        return self.spark.sql(f"SELECT * FROM {table_name}")

    def test_single_table(self):
        table_name = "spark_catalog.default.test_table1"
        comment = "comment1"
        try:
            df = self.create_and_read_table(
                table_name, [(1, "foo"), (2, "bar")], comment
            )
            tables = SparkAI._get_tables_from_explain(df)
            self.assertEqual(tables, [table_name])
            self.assertEqual(
                self.spark_ai._get_table_comment(df), "which represents comment1"
            )
        finally:
            self.spark.sql(f"DROP TABLE IF EXISTS {table_name}")

    def test_multiple_tables(self):
        table_names = [
            "spark_catalog.default.test_table1",
            "spark_catalog.default.test_table2",
        ]
        try:
            dfs = [
                self.create_and_read_table(name, [(1, "foo"), (2, "bar")])
                for name in table_names
            ]
            df = dfs[0].join(dfs[1], "col1")
            tables = SparkAI._get_tables_from_explain(df)
            self.assertEqual(tables, table_names)
            # Currently we only set the comment when reading a single table
            self.assertEqual(self.spark_ai._get_table_comment(df), "")
        finally:
            for name in table_names:
                self.spark.sql(f"DROP TABLE IF EXISTS {name}")

    def test_no_table(self):
        df = self.spark.createDataFrame([(1, "foo"), (2, "bar")], ["col1", "col2"])
        tables = SparkAI._get_tables_from_explain(df)
        self.assertEqual(tables, [])
        self.assertEqual(self.spark_ai._get_table_comment(df), "")


class SparkAITrimTextTestCase(unittest.TestCase):
    """Test cases for the _trim_text_from_end method of the SparkAI."""

    def setUp(self):
        self.llm_mock = MagicMock(spec=BaseLanguageModel)
        self.spark_ai = SparkAI(llm=self.llm_mock)

    def test_trim_text_from_end_with_text_shorter_than_max_tokens(self):
        """Tests if the function correctly returns the same text when it's shorter than the max token limit."""
        text = "This is a text"
        result = self.spark_ai._trim_text_from_end(text, max_tokens=10)
        self.assertEqual(result, text)

    def test_trim_text_from_end_with_text_longer_than_max_tokens(self):
        """Tests if the function correctly trims text to the max token limit."""
        text = "This is a longer text"
        result = self.spark_ai._trim_text_from_end(text, max_tokens=2)
        self.assertEqual(result, "This is")


class ExtractViewNameTestCase(unittest.TestCase):
    def test_extract_view_name_with_valid_create_temp_query(self):
        """Tests if the function correctly extracts the view name from a valid CREATE TEMP VIEW query"""
        query = "CREATE TEMP VIEW temp_view AS SELECT * FROM table"
        expected_view_name = "temp_view"
        actual_view_name = SparkAI._extract_view_name(query)
        self.assertEqual(actual_view_name, expected_view_name)

    def test_extract_view_name_with_valid_create_or_replace_temp_query(self):
        """Tests if the function correctly extracts the view name from a valid CREATE OR REPLACE TEMP VIEW query"""
        query = "CREATE OR REPLACE TEMP VIEW temp_view AS SELECT * FROM table"
        expected_view_name = "temp_view"
        actual_view_name = SparkAI._extract_view_name(query)
        self.assertEqual(actual_view_name, expected_view_name)

    def test_extract_view_name_with_invalid_query(self):
        """Tests if the function correctly raises a ValueError for an invalid query"""
        query = "SELECT * FROM table"
        with self.assertRaises(ValueError) as e:
            SparkAI._extract_view_name(query)
        self.assertEqual(
            str(e.exception),
            f"The provided query: '{query}' is not valid for creating a temporary view. Expected pattern: 'CREATE TEMP VIEW [VIEW_NAME] ...'",
        )

    def test_extract_view_name_with_case_insensitive_create_temp_query(self):
        """Tests if the function correctly handles case insensitivity in the CREATE TEMP VIEW keyword"""
        query = "create temp view temp_view AS SELECT * FROM table"
        expected_view_name = "temp_view"
        actual_view_name = SparkAI._extract_view_name(query)
        self.assertEqual(actual_view_name, expected_view_name)

    def test_extract_view_name_with_empty_query(self):
        """Tests if the function correctly raises a ValueError for an empty query"""
        query = ""
        with self.assertRaises(ValueError) as e:
            SparkAI._extract_view_name(query)
        self.assertEqual(
            str(e.exception),
            f"The provided query: '{query}' is not valid for creating a temporary view. Expected pattern: 'CREATE TEMP VIEW [VIEW_NAME] ...'",
        )


class URLTestCase(unittest.TestCase):
    def test_is_http_or_https_url_with_valid_http_url(self):
        """Tests if the function correctly identifies a valid http URL"""
        url = "http://www.example.com"
        result = SparkAI._is_http_or_https_url(url)
        self.assertTrue(result)

    def test_is_http_or_https_url_with_valid_https_url(self):
        """Tests if the function correctly identifies a valid https URL"""
        url = "https://www.example.com"
        result = SparkAI._is_http_or_https_url(url)
        self.assertTrue(result)

    def test_is_http_or_https_url_with_invalid_url_scheme(self):
        """Tests if the function correctly identifies an invalid URL scheme"""
        url = "ftp://www.example.com"
        result = SparkAI._is_http_or_https_url(url)
        self.assertFalse(result)

    def test_is_http_or_https_url_with_empty_string(self):
        """Tests if the function correctly identifies an empty string as an invalid URL"""
        url = ""
        result = SparkAI._is_http_or_https_url(url)
        self.assertFalse(result)

    def test_is_http_or_https_url_with_url_without_scheme(self):
        """Tests if the function correctly identifies a URL without a scheme as invalid"""
        url = "www.example.com"
        result = SparkAI._is_http_or_https_url(url)
        self.assertFalse(result)


class SparkTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm_mock = MagicMock(spec=BaseLanguageModel)
        cls.spark = (
            SparkSession.builder.master("local[*]").appName("Unit-tests").getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()


class CacheRetrievalTestCase(SparkTestCase):
    # test of cache retrieval works with a pre-populated cache

    def setUp(self):
        self.spark_ai = SparkAI(
            llm=self.llm_mock, cache_file_location="examples/spark_ai_cache.json"
        )
        self.url = (
            "https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States"
        )
        self.df1 = self.spark_ai.create_df(self.url, ["president", "vice_president"])

    def test_create_df(self):
        df2 = self.spark_ai.create_df(self.url, ["president", "vice_president"])

        assert_df_equality(self.df1, df2)

    def test_transform_df(self):
        transform_df1 = self.spark_ai.transform_df(
            self.df1, "presidents who were also vice presidents"
        )
        self.spark_ai.commit()
        transform_df2 = self.spark_ai.transform_df(
            self.df1, "presidents who were also vice presidents"
        )

        assert_df_equality(transform_df1, transform_df2)

    def test_explain_df(self):
        explain1 = self.spark_ai.explain_df(self.df1)
        self.spark_ai.commit()
        explain2 = self.spark_ai.explain_df(self.df1)

        self.assertEqual(explain1, explain2)

    def test_udf(self):
        @self.spark_ai.udf
        def convert_grades1(grade_percent: float) -> str:
            """Convert the grade percent to a letter grade using standard cutoffs"""
            ...

        self.spark_ai.commit()

        @self.spark_ai.udf
        def convert_grades2(grade_percent: float) -> str:
            """Convert the grade percent to a letter grade using standard cutoffs"""
            ...

        import random

        grade = random.randint(1, 100)

        self.assertEqual(convert_grades1(grade), convert_grades2(grade))


class SparkAnalysisTest(SparkTestCase):
    def test_analysis_handling(self):
        self.spark_ai = SparkAI(llm=self.llm_mock)
        df = self.spark.range(100).groupBy("id").count()
        left = self.spark_ai._get_analyzed_plan_from_explain(df)
        right = df._jdf.queryExecution().analyzed().toString()
        self.assertEqual(left, right)


class SparkConnectTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.remote("sc://localhost").getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()


@unittest.skip("skip until GHA secret key enabled")
class SparkConnectTests(SparkConnectTestCase):
    def setUp(self):
        self.spark_ai = SparkAI(
            cache_file_location="examples/spark_ai_cache.json", verbose=True
        )
        self.spark_ai.activate()

    def test_spark_connect_autodf_e2e(self):
        try:
            df = self.spark_ai.create_df(
                "https://www.carpro.com/blog/full-year-2022-national-auto-sales-by-brand"
            )

            # Test df aggregation happens before Pandas conversion
            self.root_logger = logging.getLogger()
            self.log_capture_string = StringIO()
            self.ch = logging.StreamHandler(self.log_capture_string)
            self.root_logger.addHandler(self.ch)
            df.ai.plot(
                "pie chart for US sales market shares, show the top 5 brands and the sum of others"
            )
            log_contents = self.log_capture_string.getvalue()
            self.root_logger.removeHandler(self.ch)
            groupby_index = log_contents.find("groupBy")
            toPandas_index = log_contents.find("toPandas")
            self.assertTrue(
                groupby_index != -1
                and toPandas_index != -1
                and groupby_index < toPandas_index,
                "the aggregation 'groupby' should appear before 'toPandas'",
            )

            df.ai.explain()
            df.ai.verify("expect all brands to be unique")
        except Exception:
            self.fail("Spark Connect auto_df end-to-end test error")

    def test_spark_connect_transform(self):
        try:
            spark = self.spark_ai._spark
            df = spark.createDataFrame(
                [
                    ("children bike", 20),
                    ("comfort bike", 15),
                    ("mountain bike", 10),
                    ("electric bike", 5),
                    ("road bike", 3),
                    ("cruisers bike", 8),
                ],
                ["product_category", "product_count"],
            )
            result = df.ai.transform("list top 3 products by count")

            expected_lst = [
                Row(product_category="children bike", product_count=20),
                Row(product_category="comfort bike", product_count=15),
                Row(product_category="mountain bike", product_count=10),
            ]

            self.assertEqual(result.collect(), expected_lst)
        except Exception:
            self.fail("Spark Connect transform error")

    def test_spark_connect_pivot(self):
        try:
            spark = self.spark_ai._spark
            df = spark.createDataFrame(
                [
                    ("A", "English", 45),
                    ("A", "Maths", 50),
                    ("B", "English", 75),
                    ("B", "Maths", 80),
                    ("C", "English", 90),
                    ("C", "Science", 100),
                ],
                ["Student", "Subject", "Marks"],
            )
            result = df.ai.transform("pivot using Subject for Marks")

            expected_lst = [
                Row(Student="B", English=75, Maths=80, Science=None),
                Row(Student="C", English=90, Maths=None, Science=100),
                Row(Student="A", English=45, Maths=50, Science=None),
            ]

            self.assertEqual(result.collect(), expected_lst)
        except Exception:
            self.fail("Spark Connect pivot error")


if __name__ == "__main__":
    unittest.main()
