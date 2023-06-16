from typing import Optional, Dict

from langchain.cache import RETURN_VAL_TYPE, SQLiteCache
from langchain.schema import Generation


class Cache:
    """
    This class provides an interface for a simple in-memory and persistent cache system. It keeps an in-memory staging
    cache, which gets updated through the `update` method and can be persisted through the `commit` method. Cache
    lookup is always performed on the persistent cache only.

    Attributes:
        _staging_updates: A dictionary to keep track of the in-memory staging updates.
        _sqlite_cache: An SQLiteCache instance that acts as the persistent cache.
    """

    def __init__(self, database_path: str = ".spark_llm.db"):
        """
        Initializes a new instance of the Cache class.

        Args:
            database_path (str, optional): The path to the database file for the SQLiteCache.
                Defaults to ".spark_llm.db".
        """
        self._staging_updates: Dict[(str, str), str] = {}
        self._sqlite_cache = SQLiteCache(database_path=database_path)

    def lookup(self, prompt: str, llm_string: str) -> Optional[str]:
        """
        Performs a lookup in the persistent cache using the given prompt and llm_string.

        Args:
            prompt (str): The prompt string for the lookup.
            llm_string (str): The LLM string for the lookup.

        Returns:
            Optional[str]: The cached text corresponding to the prompt and LLM string, if available. Otherwise, None.
        """
        lookup_result = self._sqlite_cache.lookup(prompt, llm_string)
        if lookup_result is not None and len(lookup_result) > 0:
            return lookup_result[0].text
        return None

    def update(self, prompt: str, llm_string: str, return_val: str) -> None:
        """
        Updates the staging cache with the given prompt, llm_string, and return value.

        Args:
            prompt (str): The prompt string.
            llm_string (str): The LLM string.
            return_val (str): The return value to be cached.
        """
        self._staging_updates[(prompt, llm_string)] = return_val

    def clear(self) -> None:
        """
        Clears both the in-memory staging cache and the persistent cache.
        """
        self._sqlite_cache.clear()
        self._staging_updates: Dict[(str, str), RETURN_VAL_TYPE] = {}

    def commit(self) -> None:
        """
        Commits all the staged updates to the persistent cache.
        """
        for key, value in self._staging_updates.items():
            stored_value = [Generation(text=value)]
            self._sqlite_cache.update(key[0], key[1], stored_value)
