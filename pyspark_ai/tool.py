from typing import Optional, Any, Union
from collections import OrderedDict
import os
import shutil

from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from pydantic import Field
from pyspark.sql import SparkSession
from pyspark_ai.ai_utils import AIUtils
from pyspark_ai.spark_utils import SparkUtils

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

    def _run_command(self, command: str) -> str:
        df = self.spark.sql(command)
        return str(SparkUtils.get_dataframe_results(df))

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


class LRUVectorStore:
    """Implements an LRU policy to enforce a max storage space for vector file storage."""

    def __init__(self, vector_file_dir: str, max_size: float = 16) -> None:
        # by default, max_size = 16 GB
        self.files: OrderedDict[str, float] = OrderedDict()
        self.vector_file_dir = vector_file_dir
        # represent in bytes to prevent floating point errors
        self.max_bytes = max_size * 1e9
        self.current_size = 0

        # initialize the file cache, if vector_file_dir exists
        # existing files will get evicted in reverse-alphabetical order
        # TODO: write LRU to disk, to evict existing files in LRU order
        if os.path.exists(self.vector_file_dir):
            for file in os.listdir(self.vector_file_dir):
                file_path = os.path.join(self.vector_file_dir, file)
                file_size = LRUVectorStore.get_file_size_bytes(file_path)
                if LRUVectorStore.get_file_size_bytes(file_path) <= self.max_bytes:
                    self.files[file_path] = file_size
                    self.current_size += file_size
                else:
                    shutil.rmtree(file_path)

    @staticmethod
    def get_file_size_bytes(file_path: str) -> float:
        return os.path.getsize(file_path)

    @staticmethod
    def get_storage(vector_file_dir: str) -> float:
        # calculate current storage space of vector files, in bytes
        size = 0
        for path, dirs, files in os.walk(vector_file_dir):
            for f in files:
                fp = os.path.join(path, f)
                size += LRUVectorStore.get_file_size_bytes(fp)
        return size

    def access(self, file_path: str) -> None:
        # move accessed key to end of LRU cache
        if file_path in self.files:
            self.files.move_to_end(file_path)

    def add(self, file_path: str) -> None:
        # remove file path if max storage size exceeded, else add
        curr_file_size = LRUVectorStore.get_file_size_bytes(file_path)
        if curr_file_size <= self.max_bytes:
            self.files[file_path] = curr_file_size
            self.current_size += curr_file_size
            self.files.move_to_end(file_path)
        else:
            shutil.rmtree(file_path)

        # evict files while max_size exceeded
        while self.current_size > self.max_bytes:
            evicted_file_path, evicted_file_size = self.files.popitem(last=False)
            self.current_size -= evicted_file_size
            shutil.rmtree(evicted_file_path)


class VectorSearchUtil:
    """This class contains helper methods for similarity search performed by SimilarValueTool."""

    @staticmethod
    def vector_similarity_search(
        col_lst: Optional[list],
        vector_store_path: Optional[str],
        lru_vector_store: Optional[LRUVectorStore],
        search_text: str,
    ) -> str:
        from langchain.vectorstores import FAISS
        from langchain.embeddings import HuggingFaceBgeEmbeddings

        model_name = "BAAI/bge-base-en-v1.5"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}

        if vector_store_path and os.path.exists(vector_store_path):
            vector_db = FAISS.load_local(
                vector_store_path,
                HuggingFaceBgeEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                ),
            )
            lru_vector_store.access(vector_store_path)
        else:
            vector_db = FAISS.from_texts(
                col_lst,
                HuggingFaceBgeEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                ),
            )

            if vector_store_path:
                vector_db.save_local(vector_store_path)
                lru_vector_store.add(vector_store_path)

        docs = vector_db.similarity_search(search_text)
        return docs[0].page_content


class SimilarValueTool(BaseTool):
    """Tool for finding the column value which is closest to the input text."""

    spark: Union[SparkSession, ConnectSparkSession] = Field(exclude=True)
    name = "similar_value"
    description = """
    This tool takes a string keyword and searches for the most similar value from a vector store with all
    possible values from the desired column.
    Input to this tool is a pipe-separated string in this format: keyword|column_name|temp_view_name.
    The temp_view_name will be queried in the column_name using the most similar value to the keyword.
    """

    vector_store_dir: Optional[str]
    lru_vector_store: Optional[LRUVectorStore]

    def _run(
        self, inputs: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        input_lst = inputs.split("|")

        # parse input
        search_text = input_lst[0]
        col = input_lst[1]
        temp_view_name = input_lst[2]

        vector_store_path = (
            self.vector_store_dir + temp_view_name + "_" + col
            if self.vector_store_dir
            else None
        )

        if not self.vector_store_dir or not os.path.exists(vector_store_path):
            new_df = self.spark.sql(
                "select distinct `{}` from {}".format(col, temp_view_name)
            )
            col_lst = [str(row[col]) for row in new_df.collect()]
        else:
            col_lst = None

        return VectorSearchUtil.vector_similarity_search(
            col_lst, vector_store_path, self.lru_vector_store, search_text
        )

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("SimilarityTool does not support async")
