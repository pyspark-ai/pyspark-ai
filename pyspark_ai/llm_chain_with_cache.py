from typing import Any, Optional, List

from langchain import LLMChain
from langchain.callbacks.manager import Callbacks

from pyspark_ai.cache import Cache

SKIP_CACHE_TAGS = ["SKIP_CACHE"]


class LLMChainWithCache(LLMChain):
    cache: Cache

    @staticmethod
    def _sort_and_stringify(*args: Any) -> str:
        # Convert all arguments to strings, then sort them
        sorted_args = sorted(str(arg) for arg in args)
        # Join all the sorted, stringified arguments with a space
        result = " ".join(sorted_args)
        return result

    def run(
        self,
        *args: Any,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        assert not args, "The chain expected no arguments"
        prompt_str = self.prompt.format_prompt(**kwargs).to_string()
        use_cache = tags != SKIP_CACHE_TAGS
        cached_result = (
            self.cache.lookup(prompt_str) if use_cache else None
        )
        if cached_result is not None:
            return cached_result
        result = super().run(*args, callbacks=callbacks, tags=tags, **kwargs)
        if use_cache:
            self.cache.update(prompt_str, result)
        return result
