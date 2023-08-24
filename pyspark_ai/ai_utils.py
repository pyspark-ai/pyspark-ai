import re
from typing import Optional, Type, List

from pyspark.sql import DataFrame


class AIMethodWrapper:
    """
    This class wraps the AI utility functions to allow them to be used directly
    on DataFrame instances. An instance of this class is created each time the
    utility functions are accessed, with the DataFrame and SparkAI instance
    passed to it.
    """

    def __init__(self, spark_ai, df_instance: DataFrame):
        """
        Initialize the AIMethodWrapper with the given SparkAI and DataFrame instance.

        Args:
            spark_ai: The SparkAI instance containing the AI utility methods.
            df_instance: The DataFrame instance on which the utility methods will be used.
        """
        self.spark_ai = spark_ai
        self.df_instance = df_instance

    def transform(self, desc: str, cache: bool = True) -> DataFrame:
        """
        Transform the DataFrame using the given description.

        Args:
            desc: A string description specifying the transformation.
            cache: Indicates whether to utilize a cache for this method.
                If `True`, fetches cached data, if available.
                If `False`, retrieves fresh data and updates cache.

        Returns:
            The transformed DataFrame.
        """
        return self.spark_ai.transform_df(self.df_instance, desc, cache)

    def explain(self, cache: bool = True) -> str:
        """
        Explain the DataFrame.

        Args:
            cache: Indicates whether to utilize a cache for this method.
                If `True`, fetches cached data, if available.
                If `False`, retrieves fresh data and updates cache.

        Returns:
            A string explanation of the DataFrame.

        """
        return self.spark_ai.explain_df(self.df_instance, cache)

    def plot(self, desc: Optional[str] = None, cache: bool = True) -> None:
        """
        Plot the DataFrame.

        Args:
            desc: A string description specifying the plot.
            cache: Indicates whether to utilize a cache for this method.
                If `True`, fetches cached data, if available.
                If `False`, retrieves fresh data and updates cache.
        """
        return self.spark_ai.plot_df(self.df_instance, desc, cache)

    def verify(self, desc: str, cache: bool = True) -> None:
        """
        Verify the DataFrame using the given description.

        Args:
            desc: A string description specifying what to verify in the DataFrame.
            cache: Indicates whether to utilize a cache for this method.
                If `True`, fetches cached data, if available.
                If `False`, retrieves fresh data and updates cache.
        """
        return self.spark_ai.verify_df(self.df_instance, desc, cache)


class AIUtils:
    """
    This class is a descriptor that is used to add AI utility methods to DataFrame instances.
    When the utility methods are accessed, it returns a new AIMethodWrapper instance with the
    DataFrame and SparkAI instance passed to it.
    """

    def __init__(self, spark_ai):
        """
        Initialize the AIUtils descriptor with the given SparkAI.

        Args:
            spark_ai: The SparkAI instance containing the AI utility methods.
        """
        self.spark_ai = spark_ai

    def __get__(self, instance: DataFrame, owner: Type[DataFrame]) -> AIMethodWrapper:
        """
        This method is called when the AI utility methods are accessed on a DataFrame instance.
        It returns a new AIMethodWrapper instance with the DataFrame instance and SparkAI passed to it.

        Args:
            instance: The DataFrame instance on which the utility methods are being accessed.
            owner: The class (DataFrame) to which this descriptor is added.

        Returns:
            A new AIMethodWrapper instance.
        """
        return AIMethodWrapper(self.spark_ai, instance)

    @staticmethod
    def extract_code_blocks(text: str) -> List[str]:
        # Pattern to match code blocks wrapped in triple backticks
        code_block_pattern = re.compile(r"```(.*?)```", re.DOTALL)
        triple_backtick_blocks = re.findall(code_block_pattern, text)

        extracted_blocks = []

        # Handle triple backtick blocks
        if triple_backtick_blocks:
            for block in triple_backtick_blocks:
                block = block.strip()
                if block.startswith("python"):
                    block = block.replace("python\n", "", 1)
                elif block.startswith("sql"):
                    block = block.replace("sql\n", "", 1)
                extracted_blocks.append(block)

            return extracted_blocks
        else:
            # Check for single backtick block
            if text.startswith("`") and text.endswith("`"):
                return [text.strip("`")]
            return [text]
