import re
import uuid

prefix = "spark_ai_temp_view"


def random_view_name() -> str:
    """
    Generate a random temp view name.
    """
    return f"{prefix}_{uuid.uuid4().hex[:6]}"


def canonize_string(s) -> str:
    """
    Replace all occurrences of 'spark_ai_temp_view' followed by 6 alphanumeric characters with 'spark_ai_temp_view' in a given string.

    Args:
        s (str): The string in which to replace substrings.

    Returns:
        str: The modified string with all matching substrings replaced.
    """
    pattern = f'{prefix}_[0-9a-zA-Z]{{6}}'
    return re.sub(pattern, prefix, s)
