from pyspark.sql import DataFrame
from typing import Type


class LLMMethodWrapper:

    def __init__(self, assistant, df_instance: DataFrame):
        self.assistant = assistant
        self.df_instance = df_instance

    def transform(self, desc: str) -> DataFrame:
        return self.assistant.transform_df(self.df_instance, desc)

    def explain(self) -> str:
        return self.assistant.explain_df(self.df_instance)

    def plot(self) -> None:
        return self.assistant.plot_df(self.df_instance)

    def verify(self, desc: str) -> None:
        return self.assistant.verify_df(self.df_instance, desc)


class LLMUtils:

    def __init__(self, llm_assistant):
        self.assistant = llm_assistant

    def __get__(self, instance: DataFrame, owner: Type[DataFrame]) -> LLMMethodWrapper:
        return LLMMethodWrapper(self.assistant, instance)
