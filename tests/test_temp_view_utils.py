import unittest
from pyspark_ai.temp_view_utils import random_view_name, canonize_string, replace_view_name
from pyspark.sql import SparkSession, DataFrame


class TestUtilityFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_random_view_name_dataframe(self):
        df = self.spark.createDataFrame([(1, 'foo'), (2, 'bar')], ['id', 'value'])
        view_name = random_view_name(df)
        self.assertTrue(view_name.startswith("spark_ai_temp_view"))

    def test_random_view_name_string(self):
        view_name = random_view_name("some_string")
        self.assertTrue(view_name.startswith("spark_ai_temp_view"))

    def test_random_view_name_same_logical_plan(self):
        self.spark.createDataFrame([(1, 'foo'), (2, 'bar')], ['id', 'value']).createOrReplaceTempView("temp_table")

        df1 = self.spark.sql("SELECT * FROM temp_table")
        df2 = self.spark.sql("SELECT * FROM temp_table")

        view_name1 = random_view_name(df1)
        view_name2 = random_view_name(df2)

        self.assertEqual(view_name1, view_name2)

    def test_canonize_string(self):
        original_string = "SELECT * FROM spark_ai_temp_view_123456 WHERE id = 1"
        expected_string = "SELECT * FROM spark_ai_temp_view WHERE id = 1"
        self.assertEqual(canonize_string(original_string), expected_string)

    def test_replace_view_name(self):
        original_string = "SELECT * FROM spark_ai_temp_view WHERE id = 1"
        expected_string = "SELECT * FROM spark_ai_temp_view_12345 WHERE id = 1"
        self.assertEqual(replace_view_name(original_string, "spark_ai_temp_view_12345"),
                         expected_string)