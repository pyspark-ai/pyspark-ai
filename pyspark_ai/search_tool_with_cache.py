from typing import Callable

from pyspark_ai.cache import Cache


class SearchToolWithCache:
    def __init__(self, web_search_tool: Callable[[str], str], cache: Cache):
        self.web_search_tool = web_search_tool
        self.cache = cache

    def search(self, query: str) -> str:
        # Try to get the result from the cache
        key = f"web_search:{query}"
        cached_result = self.cache.lookup(key)
        if cached_result is not None:
            return cached_result

        # If the result was not in the cache, use the web_search_tool
        result = self.web_search_tool(query)

        # Update the cache with the new result
        self.cache.update(key, result)

        return result
