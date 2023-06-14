from typing import Optional, Any, Dict

from langchain import BaseCache
from langchain.cache import RETURN_VAL_TYPE, SQLiteCache


class Cache(BaseCache):
    def __init__(self):
        self._staging_updates: Dict[(str, str), RETURN_VAL_TYPE] = {}
        self._sqlite_cache = SQLiteCache(database_path=".langchain.db")

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        return self._sqlite_cache.lookup(prompt, llm_string)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        self._staging_updates[(prompt, llm_string)] = return_val

    def clear(self, **kwargs: Any) -> None:
        self._sqlite_cache.clear()
        self._staging_updates: Dict[(str, str), RETURN_VAL_TYPE] = {}

    def commit(self) -> None:
        for (key, value) in self._staging_updates.items():
            self._sqlite_cache.update(key[0], key[1], value)