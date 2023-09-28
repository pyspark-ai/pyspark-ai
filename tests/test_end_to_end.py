import unittest

from pyspark_ai import SparkAI


class EndToEndTestCase(unittest.TestCase):
    def setUp(self):
        self.spark_ai = SparkAI()
        self.spark_ai.activate()

    def test_dataframe_transform(self):
        df = self.spark_ai._spark.createDataFrame([
            ("Alice", 1),
            ("Bob", 2),
        ], ["name", "age"])
        transformed_df = df.ai.transform("what is the name with oldest age?")
        self.assertEqual(transformed_df.collect()[0][0], "Bob")