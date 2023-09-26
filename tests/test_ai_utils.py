import unittest

from pyspark_ai.ai_utils import AIUtils


class TestExtractCodeBlocks(unittest.TestCase):

    def test_triple_backticks_with_lang(self):
        text = "```python\nprint('Hello, world!')\n```"
        self.assertEqual(AIUtils.extract_code_blocks(text), ["print('Hello, world!')"])

    def test_triple_backticks_without_lang(self):
        text = "```\nprint('Hello, world!')\n```"
        self.assertEqual(AIUtils.extract_code_blocks(text), ["print('Hello, world!')"])

    def test_single_backticks(self):
        text = "`print('Hello, world!')`"
        self.assertEqual(AIUtils.extract_code_blocks(text), ["print('Hello, world!')"])

    def test_no_backticks(self):
        text = "print('Hello, world!')"
        self.assertEqual(AIUtils.extract_code_blocks(text), ["print('Hello, world!')"])

    def test_empty_string(self):
        text = ""
        self.assertEqual(AIUtils.extract_code_blocks(text), [""])

    def test_multiple_code_blocks(self):
        text = """```python
print('Hello, world!')
        ```
        Some text here.
        ```
print('goodbye, world!')
        ```"""
        self.assertEqual(AIUtils.extract_code_blocks(text), ["print('Hello, world!')", "print('goodbye, world!')"])

    def test_sql_triple_backticks_with_lang(self):
        text = "```sql\nSELECT a FROM b;\n```"
        self.assertEqual(AIUtils.extract_code_blocks(text), ["SELECT a FROM b;"])

    def test_sql_triple_backticks_without_lang(self):
        text = "```\nSELECT a FROM b;\n```"
        self.assertEqual(AIUtils.extract_code_blocks(text), ["SELECT a FROM b;"])

    def test_sql_single_backticks(self):
        text = "`SELECT a FROM b;`"
        self.assertEqual(AIUtils.extract_code_blocks(text), ["SELECT a FROM b;"])

    def test_sql_no_backticks(self):
        text = "SELECT a FROM b;"
        self.assertEqual(AIUtils.extract_code_blocks(text), ["SELECT a FROM b;"])

    def test_multiple_sql_code_blocks(self):
        text = """```sql
SELECT a FROM b;
```
Some text here.
```
SELECT c FROM d;
```"""
        self.assertEqual(AIUtils.extract_code_blocks(text), ["SELECT a FROM b;", "SELECT c FROM d;"])


class TestRetryExecution(unittest.TestCase):

    def test_retry_execution(self):
        counter = [0]

        def failing_function():
            counter[0] += 1
            if counter[0] < 3:
                raise ValueError("This function is supposed to fail!")
            return "Success"

        try:
            result = AIUtils.retry_execution(failing_function)
        except Exception as e:
            self.fail(f"Caught exception: {e}")

        self.assertEqual(result, "Success", "Function did not succeed after retries")
        self.assertEqual(counter[0], 3, "Function was not retried the expected number of times")


if __name__ == '__main__':
    unittest.main()