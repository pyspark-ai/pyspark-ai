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
