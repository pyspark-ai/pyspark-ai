import re
import uuid

prefix = "spark_ai_temp_view"
pattern = f"{prefix}_[0-9a-zA-Z]{{6}}"


def random_view_name() -> str:
    """
    Generate a random temp view name.
    """
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


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
    return re.sub(pattern, random_view, s)
