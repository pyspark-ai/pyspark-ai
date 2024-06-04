from __future__ import annotations

import contextlib
import io
import re

from pyspark.sql import DataFrame, Row, SparkSession


class SparkUtils:
    """Class with spark helpers."""

    @staticmethod
    def _convert_row_as_tuple(row: Row) -> tuple:
        return tuple(map(str, row.asDict().values()))

    @staticmethod
    def get_dataframe_results(df: DataFrame) -> list:
        """Return rows of the DataFrame in the form of list of tuples.

        :param df: DataFrame
        :return: list of tuples
        """
        return list(map(SparkUtils._convert_row_as_tuple, df.collect()))

    @staticmethod
    def extract_view_name(query: str) -> str:
        """
        Extract the view name from the provided SQL query.

        :param query: SQL query as a string
        :return: view name as a string
        :raises: ValueError in the case when query does not create a view
        """
        pattern = r"^CREATE(?: OR REPLACE)? TEMP VIEW (\S+)"
        match = re.search(pattern, query, re.IGNORECASE)
        if not match:
            raise ValueError(
                f"The provided query: '{query}' is not valid for creating a temporary view. "
                "Expected pattern: 'CREATE TEMP VIEW [VIEW_NAME] ...'"
            )
        return match.group(1)

    @staticmethod
    def get_df_schema(df: DataFrame) -> list[str]:
        """Return DataFrame schema as list of strings.

        Output of this functions is list of strings in form 'name, dtype'

        :param df: DataFrame
        :return: list[str]
        """
        return [f"{name}, {dtype}" for name, dtype in df.dtypes]

    @staticmethod
    def get_analyzed_plan_from_explain(df: DataFrame) -> str:
        """
        Helper function to parse the content of the extended explain
        string to extract the analyzed logical plan. As Spark does not provide
        access to the logical plane without accessing the query execution object
        directly, the value is extracted from the explain text representation.

        :param df: The dataframe to extract the logical plan from.
        :return: The analyzed logical plan.
        """
        with contextlib.redirect_stdout(io.StringIO()) as f:
            df.explain(extended=True)
        explain = f.getvalue()
        splits = explain.split("\n")
        # The two index operations will fail if Spark changes the textual
        # plan representation.
        begin = splits.index("== Analyzed Logical Plan ==")
        end = splits.index("== Optimized Logical Plan ==")
        # The analyzed logical plan starts two lines after the section marker.
        # The first line is the output schema.
        return "\n".join(splits[begin + 2 : end])

    @staticmethod
    def get_sample_spark_rows(df: DataFrame, sample_rows_in_table_info) -> list:
        if sample_rows_in_table_info <= 0:
            return []

        with contextlib.suppress(Exception):
            return SparkUtils.get_dataframe_results(df.limit(3))

        # If fail to get sample rows, return empty list
        return []

    @staticmethod
    def get_tables_from_explain(df: DataFrame) -> list[str]:
        """
        Helper function to parse the tables from the content of the explanation
        """
        explain = SparkUtils.get_analyzed_plan_from_explain(df)
        splits = explain.split("\n")
        # For table relations, the table name is the 2nd element in the line
        # It can be one of the followings:
        # 1. "  +- Relation default.foo101"
        # 2. ":        +- Relation default.foo100"
        # 3. "Relation default.foo100"
        tables = []
        for line in splits:
            # if line starts with "Relation" or contains "+- Relation", it is a table relation
            if line.startswith("Relation ") or "+- Relation " in line:
                table_name_with_output = line.split("Relation ", 1)[1].split(" ")[0]
                table_name = table_name_with_output.split("[")[0]
                tables.append(table_name)

        return tables

    @staticmethod
    def get_table_comment_from_desc(table_name: str, spark: SparkSession) -> str:
        with contextlib.suppress(Exception):
            # Get the output of describe table
            outputs = spark.sql("DESC extended " + table_name).collect()
            # Get the table comment from output if the first row is "Comment"
            for row in outputs:
                if row.col_name == "Comment":
                    return row.data_type

        # If fail to get table comment, return empty string
        return ""

    @staticmethod
    def get_table_comment(df: DataFrame, spark: SparkSession) -> str:
        tables = SparkUtils.get_tables_from_explain(df)
        # To be conservative, we return the table comment if there is only one table
        if len(tables) == 1:
            comment = SparkUtils.get_table_comment_from_desc(tables[0], spark)
            if len(comment) > 0:
                return "which represents " + comment
        return ""

    @staticmethod
    def trim_hash_id(analyzed_plan: str) -> str:
        """Remove spark inner column ids from the plan representation.

        :param analyzed_plan: string representation of the analyzed logical plan
        :return: trimmed plan
        """
        # Pattern to find strings like #59 or #2021
        pattern = r"#\d+"

        # Remove matching patterns
        trimmed_plan = re.sub(pattern, "", analyzed_plan)

        return trimmed_plan
