from pyspark.sql import DataFrame
from typing import Type, Optional


class LLMMethodWrapper:
    """
    This class wraps the LLM utility functions to allow them to be used directly
    on DataFrame instances. An instance of this class is created each time the
    utility functions are accessed, with the DataFrame instance and LLM assistant
    passed to it.
    """

    def __init__(self, assistant, df_instance: DataFrame):
        """
        Initialize the LLMMethodWrapper with the given LLM assistant and DataFrame instance.

        Args:
            assistant: The SparkLLMAssistant instance containing the LLM utility methods.
            df_instance: The DataFrame instance on which the utility methods will be used.
        """
        self.assistant = assistant
        self.df_instance = df_instance

    def transform(self, desc: str) -> DataFrame:
        """
        Transform the DataFrame using the given description.

        Args:
            desc: A string description specifying the transformation.

        Returns:
            The transformed DataFrame.
        """
        return self.assistant.transform_df(self.df_instance, desc)

    def explain(self) -> str:
        """
        Explain the DataFrame.

        Returns:
            A string explanation of the DataFrame.
        """
        return self.assistant.explain_df(self.df_instance)

    def plot(self, desc: Optional[str] = None) -> None:
        """
        Plot the DataFrame.
        """
        return self.assistant.plot_df(self.df_instance, desc)

    def verify(self, desc: str) -> None:
        """
        Verify the DataFrame using the given description.

        Args:
            desc: A string description specifying what to verify in the DataFrame.
        """
        return self.assistant.verify_df(self.df_instance, desc)


class LLMUtils:
    """
    This class is a descriptor that is used to add LLM utility methods to DataFrame instances.
    When the utility methods are accessed, it returns a new LLMMethodWrapper instance with the
    DataFrame instance and LLM assistant passed to it.
    """

    def __init__(self, llm_assistant):
        """
        Initialize the LLMUtils descriptor with the given LLM assistant.

        Args:
            llm_assistant: The SparkLLMAssistant instance containing the LLM utility methods.
        """
        self.assistant = llm_assistant

    def __get__(self, instance: DataFrame, owner: Type[DataFrame]) -> LLMMethodWrapper:
        """
        This method is called when the LLM utility methods are accessed on a DataFrame instance.
        It returns a new LLMMethodWrapper instance with the DataFrame instance and LLM assistant passed to it.

        Args:
            instance: The DataFrame instance on which the utility methods are being accessed.
            owner: The class (DataFrame) to which this descriptor is added.

        Returns:
            A new LLMMethodWrapper instance.
        """
        return LLMMethodWrapper(self.assistant, instance)
