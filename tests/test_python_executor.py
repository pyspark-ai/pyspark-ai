import unittest
from typing import Any, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, ChatResult
from pyspark.sql import SparkSession
from pyspark_ai import SparkAI
from pyspark_ai.code_logger import CodeLogger
from pyspark_ai.prompt import PLOT_PROMPT
from pyspark_ai.python_executor import DataFrameLike, PythonExecutor


# Test case for PythonExecutor.
# Mock the following:
#   - llm
# For the first call of llm.predict_messages, return a response with content of "1 1", so that it will fail.
# Verify that PythonExecutor will retry and succeed.
# For the second call of llm.predict_messages, return a response with content of "1 + 1"
# Verify that PythonExecutor will succeed.
class TestPythonExecutor(unittest.TestCase):
    # Mock llm
    class MockLLM(BaseChatModel):
        predict_messages_calls: int

        def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any
        ) -> ChatResult:
            pass

        async def _agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any
        ) -> ChatResult:
            pass

        @property
        def _llm_type(self) -> str:
            pass

        def __init__(self):
            super().__init__(predict_messages_calls=0)
            self.predict_messages_calls = 0

        def predict_messages(self, messages):
            self.predict_messages_calls += 1
            if self.predict_messages_calls == 1:
                return AIMessage(content="1 1")
            return AIMessage(content="1 + 1")

    def test_retry(self):
        df = SparkSession.builder.getOrCreate().createDataFrame(
            [("1", "1")], ["input", "response"]
        )
        llm = self.MockLLM()
        executor = PythonExecutor(
            llm=llm,
            df=DataFrameLike(df=df),
            prompt=PLOT_PROMPT,
            logger=CodeLogger("test"),
        )
        executor.run(
            columns=SparkAI._get_df_schema(df),
            instruction="plot",
        )
        self.assertEqual(llm.predict_messages_calls, 2)

    def test_no_retry(self):
        df = SparkSession.builder.getOrCreate().createDataFrame(
            [("1", "1")], ["input", "response"]
        )
        llm = self.MockLLM()
        executor = PythonExecutor(
            llm=llm,
            df=DataFrameLike(df=df),
            prompt=PLOT_PROMPT,
            logger=CodeLogger("test"),
            max_retries=0,
        )
        executor.run(
            columns=SparkAI._get_df_schema(df),
            instruction="plot",
        )
        self.assertEqual(llm.predict_messages_calls, 1)
