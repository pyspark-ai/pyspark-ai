import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from langchain.cache import SQLiteCache
from langchain.schema import Generation


class FileCache(ABC):
    """Base interface for a file-based cache."""

    @abstractmethod
    def lookup(self, key: str) -> Optional[str]:
        """Perform a lookup based on the key."""

    @abstractmethod
    def update(self, key: str, val: str) -> None:
        """Update cache based on the key."""

    @abstractmethod
    def clear(self, **kwargs: Any) -> None:
        """Clear cache. Can take additional keyword arguments."""

    @abstractmethod
    def commit_staging_cache(self, staging_cache: Dict[str, str]) -> None:
        """Commit all items from the staging_cache to the cache."""


class SQLiteCacheWrapper(FileCache):
    """Wrapper class for SQLiteCache that ignores llm_string during lookups and updates."""

    def __init__(self, cache_file_location: str):
        """
        Initializes a new instance of the SQLiteCacheWrapper class.

        Args:
            cache_file_location (str): The SQLite file location
        """
        self._sqlite_cache = SQLiteCache(database_path=cache_file_location)

    def lookup(self, key: str) -> Optional[str]:
        """
        Performs a lookup in the SQLiteCache using the given key.

        Args:
            key (str): The key string for the lookup.

        Returns:
            Optional[RETURN_VAL_TYPE]: The cached value corresponding to the key, if available. Otherwise, None.
        """
        lookup_result = self._sqlite_cache.lookup(prompt=key, llm_string="")
        if lookup_result is not None and len(lookup_result) > 0:
            return lookup_result[0].text
        return None

    def update(self, key: str, val: str) -> None:
        """
        Updates the SQLiteCache with the given key and return value.

        Args:
            key (str): The key string.
            val (RETURN_VAL_TYPE): The return value to be cached.
        """
        stored_value = [Generation(text=val)]
        self._sqlite_cache.update(key, "", stored_value)

    def clear(self, **kwargs: Any) -> None:
        """
        Clears the SQLiteCache.

        Args:
            **kwargs: Additional keyword arguments for the clear method of SQLiteCache.
        """
        self._sqlite_cache.clear(**kwargs)

    def commit_staging_cache(self, staging_cache: Dict[str, str]) -> None:
        """
        Commits all items from the staging_cache to the SQLiteCache.

        Args:
            staging_cache (Dict[str, str]): The staging cache to be committed.
        """
        for key, value in staging_cache.items():
            self.update(key, value)


class JsonCache(FileCache):
    """A simple caching system using a JSON file for storage, subclass of FileCache."""

    def __init__(self, filepath: str):
        """Initialize a new JsonCache instance.

        Args:
            filepath (str): The path to the JSON file to use for the cache.
        """
        self.filepath = filepath
        # If cache file exists, load it into memory.
        self.cache = {}
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                for line in f:
                    if line.strip():  # Avoid empty lines
                        line_cache = json.loads(line)
                        self.cache[line_cache["key"]] = line_cache["value"]
        # Create an empty staging cache for storing changes before they are committed.
        self.staging_cache = {}

    def update(self, key: str, value: str) -> None:
        """Store a value in the cache for a given key.

        Args:
            key (str): The key string.
            value (RETURN_VAL_TYPE): The value to store in the cache.
        """
        # Store the value in the staging cache.
        self.staging_cache[key] = value

    def lookup(self, key: str) -> Optional[str]:
        """Retrieve a value from the cache for a given key.

        Args:
            key (str): The key string.

        Returns:
            Optional[RETURN_VAL_TYPE]: The cached value for the given key, or None if no such value exists.
        """
        return self.cache.get(key)

    def commit_staging_cache(self, staging_cache: Dict[str, str]) -> None:
        """Commit all changes in the staging cache to the cache file.

        This method writes all changes in the staging cache to the end of the cache file and then clears
        the staging cache.

        Args:
            staging_cache (Dict[str, str]): The staging cache to be committed.
        """
        # Append the staging cache to the existing cache
        self.cache.update(staging_cache)
        with open(self.filepath, "a") as f:
            for key, value in staging_cache.items():
                json.dump({"key": key, "value": value}, f)
                f.write("\n")
        # Clear the staging cache
        self.staging_cache = {}

    def clear(self, **kwargs: Any) -> None:
        """Clear the cache.

        This method removes all entries from the cache and deletes the cache file.
        """
        self.cache = {}
        self.staging_cache = {}
        os.remove(self.filepath)
