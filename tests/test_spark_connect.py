import logging
import os
import unittest
import warnings
from io import StringIO

from pyspark.sql import Row, SparkSession
from pyspark_ai import SparkAI


class SparkConnectTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "SPARK_CONNECT_MODE_ENABLE" not in os.environ:
            cls.spark = SparkSession.builder.master("local[1]").getOrCreate()
        else:
            cls.spark = SparkSession.builder.remote(
                "sc://localhost:15002"
            ).getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()


@unittest.skipUnless(
    os.environ.get("OPENAI_API_KEY") and os.environ["OPENAI_API_KEY"].strip() != "",
    "OPENAI_API_KEY is not set",
)
class SparkConnectTests(SparkConnectTestCase):
    def setUp(self):
        self.spark_ai = SparkAI(
            spark_session=self.spark,
            cache_file_location="examples/spark_ai_cache.json",
            verbose=True,
        )
        self.spark_ai.activate()

    @unittest.skipIf(
        "SPARK_CONNECT_MODE_ENABLE" not in os.environ, "not spark connect env"
    )
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

    @unittest.skipIf(
        "SPARK_CONNECT_MODE_ENABLE" not in os.environ, "not spark connect env"
    )
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

    @unittest.skipIf(
        "SPARK_CONNECT_MODE_ENABLE" not in os.environ, "not spark connect env"
    )
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ResourceWarning)
        unittest.main()
