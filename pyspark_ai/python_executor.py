import os
from typing import Any, List, Optional

from langchain.callbacks.manager import Callbacks
from langchain.chains import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, HumanMessage

from pyspark.sql import DataFrame

from pyspark_ai.ai_utils import AIUtils  # noqa: E402
from pyspark_ai.cache import Cache  # noqa: E402
from pyspark_ai.code_logger import CodeLogger  # noqa: E402
from pyspark_ai.temp_view_utils import canonize_string  # noqa: E402

SKIP_CACHE_TAGS = ["SKIP_CACHE"]


class DataFrameLike:
    def __init__(self, df: DataFrame):
        self.df = df


class PythonExecutor(LLMChain):
    """LLM Chain to generate python code. It supports caching and retrying."""

    df: DataFrameLike
    cache: Cache = None
    logger: CodeLogger = None
    max_retries: int = 3

    def run(
        self,
        *args: Any,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        assert not args, "The chain expected no arguments"
        # assert llm is an instance of BaseChatModel
        assert isinstance(
            self.llm, BaseChatModel
        ), "The llm is not an instance of BaseChatModel"
        prompt_str = canonize_string(self.prompt.format_prompt(**kwargs).to_string())
        use_cache = tags != SKIP_CACHE_TAGS
        if self.cache is not None:
            cached_result = self.cache.lookup(prompt_str) if use_cache else None
            if cached_result is not None:
                self._execute_code(self.df.df, cached_result)
                return cached_result
        messages = [HumanMessage(content=prompt_str)]
        response = self._generate_python_with_retries(
            self.df.df, self.llm, messages, self.max_retries
        )
        if use_cache and self.cache is not None:
            self.cache.update(prompt_str, response)
        return response

    @staticmethod
    def _execute_code(df: DataFrame, code: str):
        import pandas as pd  # noqa: F401

        exec(compile(code, "plot_df-CodeGen", "exec"))

    def _generate_python_with_retries(
        self,
        df: DataFrame,
        chat_model: BaseChatModel,
        messages: List[BaseMessage],
        retries: int = 3,
    ) -> str:
        response = chat_model.predict_messages(messages)
        if self.logger is not None:
            self.logger.info(response.content)
        code = "\n".join(AIUtils.extract_code_blocks(response.content))
        try:
            self._execute_code(df, code)
            return code
        except Exception as e:
            if self.logger is not None:
                self.logger.warning("Getting the following error: \n" + str(e))
            if retries <= 0:
                # if we have no more retries, raise the exception
                self.logger.info(
                    "No more retries left, please modify the instruction or modify the generated code"
                )
                return ""
            if self.logger is not None:
                self.logger.info("Retrying with " + str(retries) + " retries left")

            messages.append(response)
            # append the exception as a HumanMessage into messages
            messages.append(HumanMessage(content=str(e)))
            return self._generate_python_with_retries(
                df, chat_model, messages, retries - 1
            )
