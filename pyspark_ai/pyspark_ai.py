import contextlib
import io
import os
import re
from typing import Callable, List, Optional
from urllib.parse import urlparse

import pandas as pd  # noqa: F401
import requests
import tiktoken
from bs4 import BeautifulSoup
from langchain import BasePromptTemplate, GoogleSearchAPIWrapper, LLMChain
from langchain.agents import AgentExecutor
from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from pyspark import Row
from pyspark.sql import DataFrame, SparkSession
from tiktoken import Encoding

from pyspark_ai.ai_utils import AIUtils
from pyspark_ai.cache import Cache
from pyspark_ai.code_logger import CodeLogger
from pyspark_ai.llm_chain_with_cache import SKIP_CACHE_TAGS, LLMChainWithCache
from pyspark_ai.prompt import (
    EXPLAIN_DF_PROMPT,
    PLOT_PROMPT,
    SEARCH_PROMPT,
    SQL_PROMPT,
    UDF_PROMPT,
    VERIFY_PROMPT,
)
from pyspark_ai.react_spark_sql_agent import ReActSparkSQLAgent
from pyspark_ai.search_tool_with_cache import SearchToolWithCache
from pyspark_ai.temp_view_utils import (
    random_view_name,
    replace_view_name,
    canonize_string,
)
from pyspark_ai.tool import QuerySparkSQLTool, QueryValidationTool


class SparkAI:
    _HTTP_HEADER = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        " (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        web_search_tool: Optional[Callable[[str], str]] = None,
        spark_session: Optional[SparkSession] = None,
        enable_cache: bool = True,
        cache_file_format: str = "json",
        cache_file_location: Optional[str] = None,
        encoding: Optional[Encoding] = None,
        max_tokens_of_web_content: int = 3000,
        sample_rows_in_table_info: int = 3,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the SparkAI object with the provided parameters.

        :param llm: LLM instance for selecting web search result
                                 and writing the ingestion SQL query.
        :param web_search_tool: optional function to perform web search,
                                Google search will be used if not provided
        :param spark_session: optional SparkSession, a new one will be created if not provided
        :param encoding: optional Encoding, cl100k_base will be used if not provided
        :param max_tokens_of_web_content: maximum tokens of web content after encoding
        :param sample_rows_in_table_info: number of rows to be sampled and shown in the table info.
                                        This is only used for SQL transform. To disable it, set it to 0.
        :param verbose: whether to print out the log
        """
        self._spark = spark_session or SparkSession.builder.getOrCreate()
        if llm is None:
            llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        self._llm = llm
        self._web_search_tool = web_search_tool or self._default_web_search_tool
        if enable_cache:
            self._enable_cache = enable_cache
            if cache_file_location is not None:
                # if there is parameter setting for it, use the parameter
                self._cache_file_location = cache_file_location
            elif "AI_CACHE_FILE_LOCATION" in os.environ:
                # otherwise read from env variable AI_CACHE_FILE_LOCATION
                self._cache_file_location = os.environ["AI_CACHE_FILE_LOCATION"]
            else:
                # use default value "spark_ai_cache.json"
                self._cache_file_location = "spark_ai_cache.json"
            self._cache = Cache(
                cache_file_location=self._cache_file_location,
                file_format=cache_file_format,
            )
            self._web_search_tool = SearchToolWithCache(
                self._web_search_tool, self._cache
            ).search
        else:
            self._cache = None
        self._encoding = encoding or tiktoken.get_encoding("cl100k_base")
        self._max_tokens_of_web_content = max_tokens_of_web_content
        self._search_llm_chain = self._create_llm_chain(prompt=SEARCH_PROMPT)
        self._sql_llm_chain = self._create_llm_chain(prompt=SQL_PROMPT)
        self._explain_chain = self._create_llm_chain(prompt=EXPLAIN_DF_PROMPT)
        self._sql_agent = self._create_sql_agent()
        self._plot_chain = self._create_llm_chain(prompt=PLOT_PROMPT)
        self._verify_chain = self._create_llm_chain(prompt=VERIFY_PROMPT)
        self._udf_chain = self._create_llm_chain(prompt=UDF_PROMPT)
        self._sample_rows_in_table_info = sample_rows_in_table_info
        self._verbose = verbose
        if verbose:
            self._logger = CodeLogger("spark_ai")

    def _create_llm_chain(self, prompt: BasePromptTemplate):
        if self._cache is None:
            return LLMChain(llm=self._llm, prompt=prompt)

        return LLMChainWithCache(llm=self._llm, prompt=prompt, cache=self._cache)

    def _create_sql_agent(self):
        tools = [
            QuerySparkSQLTool(spark=self._spark),
            QueryValidationTool(spark=self._spark),
        ]
        agent = ReActSparkSQLAgent.from_llm_and_tools(
            llm=self._llm, tools=tools, verbose=True
        )
        return AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True
        )

    @staticmethod
    def _extract_view_name(query: str) -> str:
        """
        Extract the view name from the provided SQL query.

        :param query: SQL query as a string
        :return: view name as a string
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
    def _generate_search_prompt(columns: Optional[List[str]]) -> str:
        return (
            f"The best search results should contain as many as possible of these info: {','.join(columns)}"
            if columns is not None and len(columns) > 0
            else ""
        )

    @staticmethod
    def _generate_sql_prompt(columns: Optional[List[str]]) -> str:
        return (
            f"The result view MUST contain following columns: {columns}"
            if columns is not None and len(columns) > 0
            else ""
        )

    @staticmethod
    def _default_web_search_tool(desc: str) -> str:
        search_wrapper = GoogleSearchAPIWrapper()
        return str(search_wrapper.results(query=desc, num_results=10))

    @staticmethod
    def _is_http_or_https_url(s: str):
        result = urlparse(s)  # Parse the URL
        # Check if the scheme is 'http' or 'https'
        return result.scheme in ["http", "https"]

    def log(self, message: str) -> None:
        if self._verbose:
            self._logger.log(message)

    def _trim_text_from_end(self, text: str, max_tokens: int) -> str:
        """
        Trim text from the end based on the maximum number of tokens allowed.

        :param text: text to trim
        :param max_tokens: maximum tokens allowed
        :return: trimmed text
        """
        tokens = list(self._encoding.encode(text))
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return self._encoding.decode(tokens)

    def _get_url_from_search_tool(
        self, desc: str, columns: Optional[List[str]], cache: bool
    ) -> str:
        search_result = self._web_search_tool(desc)
        search_columns_hint = self._generate_search_prompt(columns)
        # Run the LLM chain to pick the best search result
        tags = self._get_tags(cache)
        return self._search_llm_chain.run(
            tags=tags,
            query=desc,
            search_results=search_result,
            columns={search_columns_hint},
        )

    def _create_dataframe_with_llm(
        self, text: str, desc: str, columns: Optional[List[str]], cache: bool
    ) -> DataFrame:
        clean_text = " ".join(text.split())
        web_content = self._trim_text_from_end(
            clean_text, self._max_tokens_of_web_content
        )

        sql_columns_hint = self._generate_sql_prompt(columns)

        # Run the LLM chain to get an ingestion SQL query
        tags = self._get_tags(cache)
        temp_view_name = random_view_name(web_content)
        llm_result = self._sql_llm_chain.run(
            tags=tags,
            query=desc,
            web_content=web_content,
            view_name=temp_view_name,
            columns=sql_columns_hint,
        )
        sql_query = AIUtils.extract_code_blocks(llm_result)[0]
        # The actual view name used in the SQL query may be different from the
        # temp view name because of caching.
        view_name = self._extract_view_name(sql_query)
        formatted_sql_query = CodeLogger.colorize_code(sql_query, "sql")
        self.log(f"SQL query for the ingestion:\n{formatted_sql_query}")
        self.log(f"Storing data into temp view: {view_name}\n")
        self._spark.sql(sql_query)
        return self._spark.table(view_name)

    def _get_df_schema(self, df: DataFrame) -> str:
        return "\n".join([f"{name}: {dtype}" for name, dtype in df.dtypes])

    @staticmethod
    def _trim_hash_id(analyzed_plan):
        # Pattern to find strings like #59 or #2021
        pattern = r"#\d+"

        # Remove matching patterns
        trimmed_plan = re.sub(pattern, "", analyzed_plan)

        return trimmed_plan

    @staticmethod
    def _get_analyzed_plan_from_explain(df: DataFrame) -> str:
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
    def _get_tables_from_explain(df: DataFrame) -> List[str]:
        """
        Helper function to parse the tables from the content of the explanation
        """
        explain = SparkAI._get_analyzed_plan_from_explain(df)
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

    def _get_df_explain(self, df: DataFrame, cache: bool) -> str:
        raw_analyzed_str = self._get_analyzed_plan_from_explain(df)
        tags = self._get_tags(cache)
        return self._explain_chain.run(
            tags=tags, input=self._trim_hash_id(raw_analyzed_str)
        )

    def _get_tags(self, cache: bool) -> Optional[List[str]]:
        if self._enable_cache and not cache:
            return SKIP_CACHE_TAGS
        return None

    def create_df(
        self, desc: str, columns: Optional[List[str]] = None, cache: bool = True
    ) -> DataFrame:
        """
        Create a Spark DataFrame by querying an LLM from web search result.

        :param desc: the description of the result DataFrame, which will be used for
                     web searching
        :param columns: the expected column names in the result DataFrame
        :param cache: If `True`, fetches cached data, if available. If `False`, retrieves fresh data and updates cache.

        :return: a Spark DataFrame
        """
        url = desc.strip()  # Remove leading and trailing whitespace
        is_url = self._is_http_or_https_url(url)
        # If the input is not a valid URL, use search tool to get the dataset.
        if not is_url:
            url = self._get_url_from_search_tool(desc, columns, cache)

        self.log(f"Parsing URL: {url}\n")
        try:
            response = requests.get(url, headers=self._HTTP_HEADER)
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            self.log(f"HTTP error occurred: {http_err}")
            return
        except Exception as err:
            self.log(f"Other error occurred: {err}")
            return

        soup = BeautifulSoup(response.text, "html.parser")

        # add url and page content to cache
        if cache:
            if self._cache.lookup(key=url):
                page_content = self._cache.lookup(key=url)
            else:
                page_content = soup.get_text()
                self._cache.update(key=url, val=page_content)
        else:
            page_content = soup.get_text()

        # If the input is a URL link, use the title of web page as the
        # dataset's description.
        if is_url:
            desc = soup.title.string
        return self._create_dataframe_with_llm(page_content, desc, columns, cache)

    def _get_transform_sql_query_from_agent(
        self,
        temp_view_name: str,
        schema: str,
        sample_rows_str: str,
        comment: str,
        desc: str,
    ) -> str:
        llm_result = self._sql_agent.run(
            view_name=temp_view_name,
            columns=schema,
            sample_rows=sample_rows_str,
            comment=comment,
            desc=desc,
        )
        sql_query_from_response = AIUtils.extract_code_blocks(llm_result)[0]
        return sql_query_from_response

    def _convert_row_as_tuple(self, row: Row) -> tuple:
        return tuple(map(str, row.asDict().values()))

    def _get_dataframe_results(self, df: DataFrame) -> list:
        return list(map(self._convert_row_as_tuple, df.collect()))

    def _get_sample_spark_rows(self, df: DataFrame, temp_view_name: str) -> str:
        if self._sample_rows_in_table_info <= 0:
            return ""
        columns_str = "\t".join([f.name for f in df.schema.fields])
        try:
            sample_rows = self._get_dataframe_results(df.limit(3))
            # save the sample rows in string format
            sample_rows_str = "\n".join(["\t".join(row) for row in sample_rows])
        except Exception:
            # If fail to get sample rows, return empty string
            sample_rows_str = ""

        return (
            "/*\n"
            f"{self._sample_rows_in_table_info} rows from {temp_view_name} table:\n"
            f"{columns_str}\n"
            f"{sample_rows_str}"
            "*/\n"
        )

    def _get_table_comment_from_desc(self, table_name: str) -> str:
        try:
            # Get the output of describe table
            outputs = self._spark.sql("DESC extended " + table_name).collect()
            # Get the table comment from output if the first row is "Comment"
            for row in outputs:
                if row.col_name == "Comment":
                    return row.data_type
            return ""
        except Exception:
            # If fail to get table comment, return empty string
            return ""

    def _get_table_comment(self, df: DataFrame) -> str:
        tables = self._get_tables_from_explain(df)
        # To be conservative, we return the table comment if there is only one table
        if len(tables) == 1:
            comment = self._get_table_comment_from_desc(tables[0])
            if len(comment) > 0:
                return "which represents " + comment
        return ""

    def _get_transform_sql_query(self, df: DataFrame, desc: str, cache: bool) -> str:
        temp_view_name = random_view_name(df)
        create_temp_view_code = CodeLogger.colorize_code(
            f'df.createOrReplaceTempView("{temp_view_name}")', "python"
        )
        self.log(f"Creating temp view for the transform:\n{create_temp_view_code}")
        df.createOrReplaceTempView(temp_view_name)
        schema_str = self._get_df_schema(df)
        sample_rows_str = self._get_sample_spark_rows(df, temp_view_name)
        comment = self._get_table_comment(df)

        if cache:
            cache_key = ReActSparkSQLAgent.cache_key(desc, schema_str)
            cached_result = self._cache.lookup(key=cache_key)
            if cached_result is not None:
                self.log("Using cached result for the transform:")
                self.log(CodeLogger.colorize_code(cached_result, "sql"))
                return replace_view_name(cached_result, temp_view_name)
            else:
                sql_query = self._get_transform_sql_query_from_agent(
                    temp_view_name, schema_str, sample_rows_str, comment, desc
                )
                self._cache.update(key=cache_key, val=canonize_string(sql_query))
                return sql_query
        else:
            return self._get_transform_sql_query_from_agent(
                temp_view_name, schema_str, sample_rows_str, comment, desc
            )

    def transform_df(self, df: DataFrame, desc: str, cache: bool = True) -> DataFrame:
        """
        This method applies a transformation to a provided Spark DataFrame,
        the specifics of which are determined by the 'desc' parameter.

        :param df: The Spark DataFrame that is to be transformed.
        :param desc: A natural language string that outlines the specific transformation to be applied on the DataFrame.
        :param cache: If `True`, fetches cached data, if available. If `False`, retrieves fresh data and updates cache.

        :return: Returns a new Spark DataFrame that is the result of applying the specified transformation
                 on the input DataFrame.
        """
        sql_query = self._get_transform_sql_query(df, desc, cache)
        return self._spark.sql(sql_query)

    def explain_df(self, df: DataFrame, cache: bool = True) -> str:
        """
        This method generates a natural language explanation of the SQL plan of the input Spark DataFrame.

        :param df: The Spark DataFrame to be explained.
        :param cache: If `True`, fetches cached data, if available. If `False`, retrieves fresh data and updates cache.

        :return: A string explanation of the DataFrame's SQL plan, detailing what the DataFrame is intended to retrieve.
        """
        explain_result = self._get_df_explain(df, cache)
        # If there is code block in the explain result, ignore it.
        if "```" in explain_result:
            summary = explain_result.split("```")[-1]
            return summary.strip()
        else:
            return explain_result

    def plot_df(
        self, df: DataFrame, desc: Optional[str] = None, cache: bool = True
    ) -> None:
        instruction = f"The purpose of the plot: {desc}" if desc is not None else ""
        tags = self._get_tags(cache)
        response = self._plot_chain.run(
            tags=tags,
            columns=self._get_df_schema(df),
            explain=self._get_df_explain(df, cache),
            instruction=instruction,
        )
        self.log(response)
        codeblocks = AIUtils.extract_code_blocks(response)
        code = "\n".join(codeblocks)
        try:
            exec(compile(code, "plot_df-CodeGen", "exec"))
        except Exception as e:
            raise Exception("Could not evaluate Python code", e)

    def verify_df(self, df: DataFrame, desc: str, cache: bool = True) -> None:
        """
        This method creates and runs test cases for the provided PySpark dataframe transformation function.

        :param df: The Spark DataFrame to be verified
        :param desc: A description of the expectation to be verified
        :param cache: If `True`, fetches cached data, if available. If `False`, retrieves fresh data and updates cache.
        """
        tags = self._get_tags(cache)
        llm_output = self._verify_chain.run(tags=tags, df=df, desc=desc)

        codeblocks = AIUtils.extract_code_blocks(llm_output)
        llm_output = "\n".join(codeblocks)

        self.log(f"LLM Output:\n{llm_output}")

        formatted_code = CodeLogger.colorize_code(llm_output, "python")
        self.log(f"Generated code:\n{formatted_code}")

        locals_ = {}
        try:
            exec(compile(llm_output, "verify_df-CodeGen", "exec"), {"df": df}, locals_)
        except Exception as e:
            raise Exception("Could not evaluate Python code", e)
        self.log(f"\nResult: {locals_['result']}")

    def udf(self, func: Callable) -> Callable:
        from inspect import signature

        desc = func.__doc__
        func_signature = str(signature(func))
        input_args_types = func_signature.split("->")[0].strip()
        return_type = func_signature.split("->")[1].strip()
        udf_name = func.__name__

        code = self._udf_chain.run(
            input_args_types=input_args_types,
            desc=desc,
            return_type=return_type,
            udf_name=udf_name,
        )

        formatted_code = CodeLogger.colorize_code(code, "python")
        self.log(f"Creating following Python UDF:\n{formatted_code}")

        locals_ = {}
        try:
            exec(compile(code, "udf-CodeGen", "exec"), globals(), locals_)
        except Exception as e:
            raise Exception("Could not evaluate Python code", e)
        return locals_[udf_name]

    def activate(self):
        """
        Activates AI utility functions for Spark DataFrame.
        """
        DataFrame.ai = AIUtils(self)
        # Patch the Spark Connect DataFrame as well.
        try:
            from pyspark.sql.connect.dataframe import DataFrame as CDataFrame

            CDataFrame.ai = AIUtils(self)
        except ImportError:
            self.log(
                "The pyspark.sql.connect.dataframe module could not be imported. "
                "This might be due to your PySpark version being below 3.4."
            )

    def commit(self):
        """
        Commit the staging in-memory cache into persistent cache, if cache is enabled.
        """
        if self._cache is not None:
            self._cache.commit()
