# Cache

The English SDK supports a simple in-memory and persistent cache system. It keeps an in-memory staging cache, which gets updated for LLM and web search results. The staging cache can be persisted through the commit() method. Cache lookup is always performed on both in-memory staging cache and persistent cache.
```python
spark_ai.commit()
```