from pyspark.sql import DataFrame, Row


class SparkUtils:
    @staticmethod
    def _convert_row_as_tuple(row: Row) -> tuple:
        return tuple(map(str, row.asDict().values()))

    @staticmethod
    def get_dataframe_results(df: DataFrame) -> list:
        return list(map(SparkUtils._convert_row_as_tuple, df.collect()))
