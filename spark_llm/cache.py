from typing import Optional, Dict

from langchain.cache import RETURN_VAL_TYPE, SQLiteCache
from langchain.schema import Generation


class Cache:
    def __init__(self, database_path: str = ".spark_llm.db"):
        self._staging_updates: Dict[(str, str), str] = {}
        self._sqlite_cache = SQLiteCache(database_path=database_path)

    def lookup(self, prompt: str, llm_string: str) -> Optional[str]:
        lookup_result = self._sqlite_cache.lookup(prompt, llm_string)
        if lookup_result is not None and len(lookup_result) > 0:
            return lookup_result[0].text
        return None

    def update(self, prompt: str, llm_string: str, return_val: str) -> None:
        self._staging_updates[(prompt, llm_string)] = return_val

    def clear(self) -> None:
        self._sqlite_cache.clear()
        self._staging_updates: Dict[(str, str), RETURN_VAL_TYPE] = {}

    def commit(self) -> None:
        for (key, value) in self._staging_updates.items():
            stored_value = [Generation(text=value)]
            self._sqlite_cache.update(key[0], key[1], stored_value)