import json
import logging
import unittest
from io import StringIO
from random import random, shuffle
from typing import List
from unittest.mock import MagicMock

import numpy as np
from chispa.dataframe_comparer import assert_df_equality
from langchain.base_language import BaseLanguageModel
from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import expr, lit, mean, stddev
from pyspark.sql.types import ArrayType, StringType, DoubleType
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


class UDFGenerationTest(SparkTestCase):
    # end2end test

    def setUp(self):
        self.spark_ai = SparkAI(
            cache_file_location="examples/spark_ai_cache.json",
            verbose=True,
        )

        # Generate heterogeneous JSON by randomly reorder list of keys and drop some of them
        random_dict = {
            "id": 1279,
            "first_name": "John",
            "last_name": "Doe",
            "username": "johndoe",
            "email": "john_doe@example.com",
            "phone_number": "+1 234 567 8900",
            "address": "123 Main St, Springfield, OH, 45503, USA",
            "age": 32,
            "registration_date": "2020-01-20T12:12:12Z",
            "last_login": "2022-03-21T07:25:34Z",
        }
        self.json_keys = list(random_dict.keys())
        rows = []
        for _ in range(20):
            keys = [k for k in random_dict.keys()]
            shuffle(keys)
            rows.append({k: random_dict[k] for k in keys if random() <= 0.6})

        self.bad_json = self.spark.createDataFrame(
            [(json.dumps(val), self.json_keys) for val in rows],
            ["json_field", "schema"],
        )
        # Generate expected output of parsed JSON: list of fields in a right order or null if field is missing
        self.expected_output = self.spark.createDataFrame(
            [
                (
                    json.dumps(val),
                    self.json_keys,
                    [val.get(k, None) for k in self.json_keys],
                )
                for val in rows
            ],
            ["json_field", "schema", "parsed"],
        )

        # Add a DataFrame with random texts that contain emails
        self.email_df = self.spark.createDataFrame(
            [
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed egestas nulla sit amet elit volutpat ultricies. Morbi lacinia est fringilla pulvinar elementum. Curabitur rhoncus luctus dui, sodales blandit arcu maximus a. Aenean iaculis nulla ac enim tincidunt, et tristique enim bibendum. Fusce mollis nibh sit amet nisi pellentesque egestas. Quisque volutpat, neque eu semper tristique, odio nunc auctor odio, at condimentum lorem nunc nec nisi. Quisque auctor at velit nec fermentum. Nunc id pellentesque erat, et dignissim felis. ali.brown@gmail.com Suspendisse potenti. Donec tincidunt enim in ipsum faucibus sollicitudin. Sed placerat tempor eros at blandit. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Donec aliquam velit vehicula massa egestas faucibus. Ut pulvinar mi id pretium dignissim. Phasellus vehicula, dui sit amet porttitor effectively maximizes an attacker's chance to obtain valid credentials. Sed malesuada justo enim, et interdum mauris ullamcorper ac.",
                "Vestibulum rhoncus magna semper est lobortis gravida. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. In hac habitasse platea dictumst. michael.hobb@gmail.com Aenean sapien magna, consequat vitae pretium ac, gravida sit amet nibh. Maecenas lacinia orci in luctus placerat. Praesent lobortis turpis nec risus dapibus, eget ornare mi egestas. Nam eget dui ac mi laoreet sagittis. Integer condimentum est ac velit volutpat pharetra. Nulla facilisi. Nunc euismod, neque vitae porttitor maximus, justo dui efficitur ligula, vitae tincidunt erat neque ac nibh. Duis eu dui in erat blandit mattis.",
                "Aenean vitae iaculis odio. Donec laoreet non urna sed posuere. Nulla vitae orci finibus, convallis mauris nec, mattis augue. Proin bibendum non justo non scelerisque. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Aenean scott_p@ymail.com adipiscing diam eget ultrices ultricies. Aliquam bibendum dolor vel orci posuere, sed pulvinar enim rutrum. Nulla facilisi. Sed cursus justo sed velit pharetra auctor. Suspendisse facilisis nibh id nibh ultrices luctus.",
                "Quisque varius erat sed leo ornare, et elementum leo interdum. Aliquam erat volutpat. Ut laoreet tempus elit quis venenatis. Integer porta, lorem ut pretium luctus, erika.23@hotmail.com quis ipsum facilisis, feugiat libero sed, malesuada augue. Fusce id elementum sapien, sed SC ingeniously maximizes the chance to obtain valid credentials. Nullam imperdiet felis in metus semper ultrices. Integer vel quam consectetur, lobortis est vitae, lobortis sem. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos.",
                "Sed consectetur nisl quis mauris laoreet posuere. Phasellus in elementum orci, vitae auctor dui. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Donec eleifend mauris id auctor blandit. john.smith@protonmail.com Integer quis justo non eros convallis aliquet cursus eu dolor. Praesent nec sem a massa facilisis consectetur. Nunc pharetra sapien non erat semper, ut tempus risus vulputate. Donec lacinia condimentum arcu, ac molestie metus interdum in. Duis arcu quam, hendrerit quis venenatis sed, porta at erat.",
            ],
            schema="string",
        )
        self.parsed_email_df = self.spark.createDataFrame(
            [
                "ali.brown@gmail.com",
                "michael.hobb@gmail.com",
                "scott_p@ymail.com",
                "erika.23@hotmail.com",
                "john.smith@protonmail.com",
            ],
            schema="string",
        )

    def test_array_udf_output(self):
        @self.spark_ai.udf
        def parse_heterogeneous_json(json_str: str, schema: List[str]) -> List[str]:
            """Extract fields from heterogeneous JSON string based on given schema in a right order. If field is missing replace it by None. All imports should be inside function."""
            ...

        self.spark.udf.register(
            "parse_heterogeneous_json",
            parse_heterogeneous_json,
            returnType=ArrayType(elementType=StringType()),
        )
        res = self.bad_json.withColumn(
            "parsed", expr("parse_heterogeneous_json(json_field, schema)")
        )
        assert_df_equality(res, self.expected_output)

    def test_parse_email_udf(self):
        @self.spark_ai.udf
        def extract_email(text: str) -> str:
            """Extract first email from raw text"""
            ...

        self.spark.udf.register("extract_email", extract_email)
        assert_df_equality(
            self.email_df.withColumn("value", expr("extract_email(value)")),
            self.parsed_email_df,
        )

    def test_laplace_random_udf(self):
        @self.spark_ai.udf
        def laplace_random_number(loc: float, scale: float) -> float:
            """Generate a random number from Laplace distribution with given loc and scale in pure Python. Function should contain all necessary imports."""
            ...

        numpy_results = np.random.laplace(1.0, 0.3, 500_000)
        numpy_mean = numpy_results.mean()
        numpy_stddev = numpy_results.std()

        self.spark.udf.register(
            "laplace_random_number", laplace_random_number, returnType=DoubleType()
        )
        spark_results = (
            self.spark.sparkContext.range(0, 500_000)
            .toDF(schema="int")
            .withColumn("loc", lit(1.0).cast("double"))
            .withColumn("scale", lit(0.3).cast("double"))
            .withColumn("laplace_random", expr("laplace_random_number(loc, scale)"))
            .select(
                mean("laplace_random").alias("mean"),
                stddev("laplace_random").alias("stddev"),
            )
            .collect()
        )[0]

        assert abs(spark_results["mean"] - numpy_mean) / numpy_mean <= 0.05
        assert abs(spark_results["stddev"] - numpy_stddev) / numpy_stddev <= 0.05


if __name__ == "__main__":
    unittest.main()
