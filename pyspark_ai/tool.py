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
    Input should be pipe-separated, in the format: keyword|temp_view_name
    """

    def _run(
        self, input: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        input_lst = input.split("|")

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
    This tool finds the semantically closest word to a keyword from a vector database, using the FAISS library.
    Input to this tool is a pipe-separated string in this format: keyword|column_name|temp_view_name.
    The temp_view_name will be queried in the column_name for the semantically closest word to the keyword.
    """

    def vector_similarity_search(self, search_text: str, col: str, temp_name: str):
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
        from collections import defaultdict
        import dill

        df = self.spark.sql("select * from {}".format(temp_name))

        col_index = df.columns.index(col)

        dict_object = None

        try:
            with open('data/indices.pkl', 'rb') as openfile:
                # read from pkl file
                dict_object = dill.load(openfile)
                print("open file")
        except Exception as e:
            print(e)

        vectors_dict = dict_object if dict_object else defaultdict(dict)

        encoder = SentenceTransformer("paraphrase-mpnet-base-v2")

        if temp_name in vectors_dict.keys() and col in vectors_dict[temp_name].keys():
            vectors = vectors_dict[temp_name][col]
        else:
            col_lst = [str(x) for x in df.rdd.map(lambda x: x[col_index]).collect()]
            vectors = encoder.encode(col_lst)

            # store in dict
            vectors_dict[temp_name][col] = vectors

            # write to json
            with open("data/indices.pkl", "wb") as outfile:
                dill.dump(vectors_dict, outfile)

        # build faiss index from vectors
        vector_dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(vector_dimension)
        faiss.normalize_L2(vectors)
        index.add(vectors)

        # create search vector
        search_vector = encoder.encode(str(search_text))
        _vector = np.array([search_vector])
        faiss.normalize_L2(_vector)

        # search
        k = index.ntotal
        distances, ann = index.search(_vector, k=k)

        # get result
        result = df.toPandas()[col][ann[0][0]]
        return result

    def _run(
        self, input: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        input_lst = input.split("|")

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
