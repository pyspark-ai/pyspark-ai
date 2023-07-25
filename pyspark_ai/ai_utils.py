from pyspark.sql import DataFrame
from typing import Type, Optional, List, Iterable

class AIHistoryElement:

    def __init__(self, prompt: str, llm_result: str, df: DataFrame) -> None:
        self._prompt = prompt
        self._llm_result = llm_result
        self._df = df

    @property
    def prompt(self):
        return self._prompt

    @property
    def llm_result(self):
        return self._llm_result

    @property
    def df(self):
        return self._df

    def __repr__(self) -> str:
        return f"<AIHistoryElement prompt: {self._prompt}>"

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
        self._history: List[AIHistoryElement] = []

    def transform(self, desc: str, cache: bool = True, language: str = 'SQL') -> DataFrame:
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
        return self.spark_ai.transform_df(self.df_instance, desc=desc, cache=cache, language=language)

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

    def history(self) -> Iterable[AIHistoryElement]:
        return self._history

    def push_all(self, *history: AIHistoryElement) -> None:
        for h in history:
            self._history.append(h)

    def push(self, prompt: str, llm_result: str, df: DataFrame) -> None:
        self._history.append(AIHistoryElement(prompt, llm_result, df))



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
        try:
            # This will trigger an AttributeError in PySpark, but yield a colum in
            # Spark Connect.
            ai = instance._ai
            try:
                # Spark Connect specific behavior below:
                from pyspark.sql.connect.column import Column as CColumn
                if type(ai) == CColumn:
                    instance._ai = AIMethodWrapper(self.spark_ai, instance)
                ai = instance._ai
            except ImportError as e:
                print(e)
                # If Spark Connect is not available, simply ignore this error.
                pass
            return ai
        except AttributeError:
            instance._ai = AIMethodWrapper(self.spark_ai, instance)
            return instance._ai
