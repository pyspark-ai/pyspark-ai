import os
import json
import unittest
from random import random, shuffle
from typing import List

import numpy as np
from chispa.dataframe_comparer import assert_df_equality
from langchain.chat_models import ChatOpenAI
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, lit, mean, stddev
from pyspark.sql.types import ArrayType, DoubleType, StringType
from pyspark_ai import SparkAI
from benchmark.wikisql.wiki_sql import (
    get_table_name,
    create_temp_view_statements,
)


@unittest.skipUnless(
    os.environ.get("OPENAI_API_KEY") and os.environ["OPENAI_API_KEY"].strip() != "",
    "OPENAI_API_KEY is not set",
)
class EndToEndTestCaseGPT35(unittest.TestCase):
    def setup_spark_ai(self, spark: SparkSession):
        self.spark_ai = SparkAI(llm=ChatOpenAI(model_name="gpt-3.5-turbo"))
        self.spark_ai.activate()

    def setUp(self):
        self.spark = (
            SparkSession.builder.master("local[*]")
            .appName("End2end-tests")
            .getOrCreate()
        )
        self.setup_spark_ai(self.spark)

    def create_and_get_table_name(
        self, table: str, table_file: str = "tests/data/test_transform_e2e.tables.jsonl"
    ):
        """Util function to create temp view for desired table and return the table name"""
        statements = create_temp_view_statements(table_file)
        tbl_in_json = table.replace("-", "_")

        for statement in statements:
            if tbl_in_json in statement:
                self.spark.sql(statement)

        return get_table_name(table)

    def test_dataframe_transform(self):
        df = self.spark_ai._spark.createDataFrame(
            [
                ("Alice", 1),
                ("Bob", 2),
            ],
            ["name", "age"],
        )
        transformed_df = df.ai.transform("what is the name with oldest age?")
        self.assertEqual(transformed_df.collect()[0][0], "Bob")

    def test_transform_col_query_nondescriptive(self):
        """Test that agent selects correct query column, even with non-descriptive column names,
        by using sample column values"""
        df = self.spark_ai._spark.createDataFrame(
            [
                ("Shanghai", 31, "China"),
                ("Seattle", 30, "United States"),
                ("Austin", 33, "United States"),
                ("Paris", 29, "France"),
            ],
            ["col1", "col2", "col3"],
        )
        transformed_df = df.ai.transform("what city had the warmest temperature?")
        self.assertEqual(transformed_df.collect()[0][0], "Austin")

    def test_plot(self):
        flight_df = self.spark_ai._spark.read.option("header", "true").csv("tests/data/2011_february_aa_flight_paths.csv")
        # The following plotting will probably fail on the first run with error:
        #     'DataFrame' object has no attribute 'date'
        code = flight_df.ai.plot("Boxplot summarizing the range of starting latitudes for all AA flights in February 2011.")
        assert(code != "")


class EndToEndTestCaseGPT4(EndToEndTestCaseGPT35):
    def setup_spark_ai(self, spark: SparkSession):
        self.spark_ai = SparkAI(llm=ChatOpenAI(model_name="gpt-4"))
        self.spark_ai.activate()

    def setUp(self):
        super().setUp()
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

    def test_transform_col_query_wikisql(self):
        """Test that agent selects correct query column for ambiguous wikisql table example"""
        table_name = self.create_and_get_table_name("1-1108394-47")

        try:
            df = self.spark.table(f"`{table_name}`")
            df.createOrReplaceTempView(f"`{table_name}`")

            transformed_df = df.ai.transform(
                "which candidate won 88 votes in Queens in 1921?"
            )
            self.assertEqual(transformed_df.collect()[0][0], "jerome t. de hunt")
        finally:
            self.spark.sql(f"DROP TABLE IF EXISTS {table_name}")

    def test_filter_exact(self):
        """Test that agent filters by an exact value"""
        table_name = self.create_and_get_table_name("1-11545282-10")

        try:
            df = self.spark.table(f"`{table_name}`")
            df.createOrReplaceTempView(f"`{table_name}`")

            transformed_df = df.ai.transform("which forward player has the number 22?")
            self.assertEqual(transformed_df.collect()[0][0], "henry james")
        finally:
            self.spark.sql(f"DROP TABLE IF EXISTS {table_name}")

    # The following tests are skipped for gpt-3.5-turbo because they can be flaky
    # Also, our current focus is on DataFrame transform and plotting.
    @unittest.skip("skip test due to nondeterministic behavior")
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

    @unittest.skip("skip test due to nondeterministic behavior")
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