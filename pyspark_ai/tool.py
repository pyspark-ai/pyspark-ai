from typing import Optional, Any, Union

from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from pydantic import Field
from pyspark import Row
from pyspark.sql import SparkSession, DataFrame
from pyspark_ai.ai_utils import AIUtils

try:
    from pyspark.sql.connect.session import SparkSession as ConnectSparkSession
except ImportError:
    # For Spark version < 3.4.0, the SparkSession class is in the pyspark.sql.session module
    ConnectSparkSession = SparkSession


class QuerySparkSQLTool(BaseTool):
    """Tool for querying a Spark SQL."""

    spark: Union[SparkSession, ConnectSparkSession] = Field(exclude=True)
    name = "query_sql_db"
    description = """
        Input to this tool is a detailed and correct SQL query, output is a result from the Spark SQL.
        If the query is not correct, an error message will be returned.
        If an error is returned, rewrite the query, check the query, and try again.
        """

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the query, return the results or an error message."""
        return self._run_no_throw(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("QuerySqlDbTool does not support async")

    def _convert_row_as_tuple(self, row: Row) -> tuple:
        return tuple(map(str, row.asDict().values()))

    def _get_dataframe_results(self, df: DataFrame) -> list:
        return list(map(self._convert_row_as_tuple, df.collect()))

    def _run_command(self, command: str) -> str:
        df = self.spark.sql(command)
        return str(self._get_dataframe_results(df))

    def _run_no_throw(self, command: str) -> str:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.

        If the statement throws an error, the error message is returned.
        """
        try:
            from pyspark.errors import PySparkException
        except ImportError:
            raise ValueError(
                "pyspark is not installed. Please install it with `pip install pyspark`"
            )
        try:
            return self._run_command(command)
        except PySparkException as e:
            """Format the error message"""
            return f"Error: {e}"


class QueryValidationTool(BaseTool):
    """Tool for validating a Spark SQL query."""

    spark: Union[SparkSession, ConnectSparkSession] = Field(exclude=True)
    name = "query_validation"
    description = """
    Use this tool to double check if your query is correct before returning it.
    Always use this tool before returning a query as answer!
    """

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        try:
            from pyspark.errors import PySparkException
        except ImportError:
            raise ValueError(
                "pyspark is not installed. Please install it with `pip install pyspark`"
            )
        try:
            # The generated query from LLM can contain backticks, which are not supported by Spark SQL.
            actual_query = AIUtils.extract_code_blocks(query)[0]
            self.spark.sql(actual_query)
            return "OK"
        except PySparkException as e:
            """Format the error message"""
            return f"Error: {e}"

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("ListTablesSqlDbTool does not support async")


class ColumnQueryTool(BaseTool):
    """Tool for finding the correct column name given keywords from a question."""

    spark: Union[SparkSession, ConnectSparkSession] = Field(exclude=True)
    name = "get_column_name"
    description = """
    This tool determines which column contains a keyword from the question.
    Input to this tool is a str with a keyword of interest and a temp view name, output is the column name from the df 
    that contains the keyword.
    Input should be comma-separated, in the format: keyword,temp_view_name
    """

    def _run(
        self, input: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        input_lst = input.split(",")

        keyword = input_lst[0].lower()
        temp_name = input_lst[1]

        # get columns
        df = self.spark.sql("select * from {}".format(temp_name))
        col_lst = df.columns

        for col in col_lst:
            result = self.spark.sql(
                "select * from {} where `{}` like '%{}%'".format(
                    temp_name, col, keyword
                )
            )

            if len(result.collect()) != 0:
                return col

        return ""

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("ColumnQueryTool does not support async")


class SimilarValueTool(BaseTool):
    """Tool for running a Spark SQL query to rank the values in a column by similarity score to a keyword."""

    spark: Union[SparkSession, ConnectSparkSession] = Field(exclude=True)
    name = "similar_value"
    description = """
    Input to this tool is a SQL query that applies a UDF, similarity(col, keyword), to get similarity scores for
    all values in column col and rank the dataframe by similarity scores of the values.
    """

    def _run(
        self, command: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        df = self.spark.sql(command.lower())
        return str(self._get_dataframe_results(df))

    def _get_dataframe_results(self, df: DataFrame) -> list:
        return list(map(self._convert_row_as_tuple, df.collect()))

    def _convert_row_as_tuple(self, row: Row) -> tuple:
        return tuple(map(str, row.asDict().values()))

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("SimilarityTool does not support async")
