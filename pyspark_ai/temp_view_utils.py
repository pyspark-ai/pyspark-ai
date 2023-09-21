import re

from pyspark.sql import DataFrame

prefix = "spark_ai_temp_view"
pattern = f"{prefix}__?[0-9]*"


def random_view_name(obj) -> str:
    """
    Generate a random temp view name.
    """
    if isinstance(obj, DataFrame):
        hashcode = hash(obj.semanticHash())
    else:
        hashcode = hash(obj) % 1000000
    hash_code_str = str(hashcode).replace("-", "_")
    return f"{prefix}_{hash_code_str}"


def canonize_string(s: str) -> str:
    """
    Replace all occurrences of 'spark_ai_temp_view' followed by 6 alphanumeric characters with 'spark_ai_temp_view'
     in a given string.

    Args:
        s (str): The string in which to replace substrings.

    Returns:
        str: The modified string with all matching substrings replaced.
    """
    return re.sub(pattern, prefix, s)


def replace_view_name(s: str, random_view: str) -> str:
    """
    Replace all the 'spark_ai_temp_view' followed by 6 alphanumeric characters in a given string with a random view
     name.
    """
    return re.sub(prefix, random_view, s)
