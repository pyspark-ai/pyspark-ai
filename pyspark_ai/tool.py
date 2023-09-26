from typing import Optional, Any, Union, List

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


class SimilarValueTool(BaseTool):
    """Tool for finding the semantically closest word to a keyword from a vector database."""

    spark: Union[SparkSession, ConnectSparkSession] = Field(exclude=True)
    name = "similar_value"
    description = """
    This tool finds the semantically closest word to a keyword from a vector database, using the FAISS library.
    Input to this tool is a pipe-separated string in this format: keyword|column_name|temp_view_name.
    The temp_view_name will be queried in the column_name for the semantically closest word to the keyword.
    """

    vector_store_path: Optional[str]


    def vector_similarity_search(self, search_text: str, col: str, temp_name: str):
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import semantic_search
        from collections import defaultdict
        import dill

        new_df = self.spark.sql("select distinct `{}` from {}".format(col, temp_name))
        new_df_lst = [str(row[col]) for row in new_df.collect()]

        dict_object = None

        if self.vector_store_path:
            try:
                with open(self.vector_store_path, 'rb') as openfile:
                    # read from pkl file
                    dict_object = dill.load(openfile)
                    print("open file")
            except Exception as e:
                print(e)
                pass

        indices_dict = dict_object if dict_object else defaultdict(dict)

        encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

        if temp_name in indices_dict.keys() and col in indices_dict[temp_name].keys():
            embeddings_1 = indices_dict[temp_name][col]
        else:
            embeddings_1 = encoder.encode(new_df_lst)

            # store in dict
            indices_dict[temp_name][col] = embeddings_1

            # write to json
            with open(self.vector_store_path, "wb") as outfile:
                dill.dump(indices_dict, outfile)

        search_lst = [search_text]
        embeddings_2 = encoder.encode(search_lst, normalize_embeddings=True)
        hits = semantic_search(embeddings_2, embeddings_1, top_k=5)
        top_5_lst = [new_df_lst[hits[0][i]['corpus_id']] for i in range(len(hits[0]))]
        return top_5_lst[0]


    def _run(
        self, inputs: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        input_lst = inputs.split("|")

        search_text = input_lst[0]
        col = input_lst[1]
        temp_name = input_lst[2]

        return self.vector_similarity_search(search_text, col, temp_name)

    def _get_dataframe_results(self, df: DataFrame) -> list:
        return list(map(self._convert_row_as_tuple, df.collect()))

    def _convert_row_as_tuple(self, row: Row) -> tuple:
        return tuple(map(str, row.asDict().values()))

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("SimilarityTool does not support async")
