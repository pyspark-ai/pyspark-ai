import unittest
from unittest.mock import MagicMock

from chispa.dataframe_comparer import assert_df_equality
from langchain.base_language import BaseLanguageModel
from tiktoken import Encoding

from spark_llm import SparkLLMAssistant
from spark_llm.search_tool_with_cache import SearchToolWithCache
from pyspark.sql import SparkSession


class SparkLLMAssistantInitializationTestCase(unittest.TestCase):
    """Test cases for SparkLLMAssistant initialization."""

    def setUp(self):
        self.llm_mock = MagicMock(spec=BaseLanguageModel)
        self.spark_session_mock = MagicMock(spec=SparkSession)
        self.encoding_mock = MagicMock(spec=Encoding)
        self.web_search_tool_mock = MagicMock(spec=SearchToolWithCache.search)
        self.assistant = SparkLLMAssistant(
            llm=self.llm_mock,
            web_search_tool=self.web_search_tool_mock,
            spark_session=self.spark_session_mock,
            enable_cache=False,
            encoding=self.encoding_mock,
        )

    def test_initialization_with_default_values(self):
        """Tests if the class initializes correctly with default values."""
        self.assertEqual(self.assistant._spark, self.spark_session_mock)
        self.assertEqual(self.assistant._llm, self.llm_mock)
        self.assertEqual(
            self.assistant._web_search_tool, self.web_search_tool_mock
        )
        self.assertEqual(self.assistant._encoding, self.encoding_mock)
        self.assertEqual(self.assistant._max_tokens_of_web_content, 3000)


class SparkLLMAssistantTrimTextTestCase(unittest.TestCase):
    """Test cases for the _trim_text_from_end method of the SparkLLMAssistant."""

    def setUp(self):
        self.llm_mock = MagicMock(spec=BaseLanguageModel)
        self.assistant = SparkLLMAssistant(llm=self.llm_mock)

    def test_trim_text_from_end_with_text_shorter_than_max_tokens(self):
        """Tests if the function correctly returns the same text when it's shorter than the max token limit."""
        text = "This is a text"
        result = self.assistant._trim_text_from_end(text, max_tokens=10)
        self.assertEqual(result, text)

    def test_trim_text_from_end_with_text_longer_than_max_tokens(self):
        """Tests if the function correctly trims text to the max token limit."""
        text = "This is a longer text"
        result = self.assistant._trim_text_from_end(text, max_tokens=2)
        self.assertEqual(result, "This is")


class ExtractViewNameTestCase(unittest.TestCase):
    def test_extract_view_name_with_valid_create_temp_query(self):
        """Tests if the function correctly extracts the view name from a valid CREATE TEMP VIEW query"""
        query = "CREATE TEMP VIEW temp_view AS SELECT * FROM table"
        expected_view_name = "temp_view"
        actual_view_name = SparkLLMAssistant._extract_view_name(query)
        self.assertEqual(actual_view_name, expected_view_name)

    def test_extract_view_name_with_valid_create_or_replace_temp_query(self):
        """Tests if the function correctly extracts the view name from a valid CREATE OR REPLACE TEMP VIEW query"""
        query = "CREATE OR REPLACE TEMP VIEW temp_view AS SELECT * FROM table"
        expected_view_name = "temp_view"
        actual_view_name = SparkLLMAssistant._extract_view_name(query)
        self.assertEqual(actual_view_name, expected_view_name)

    def test_extract_view_name_with_invalid_query(self):
        """Tests if the function correctly raises a ValueError for an invalid query"""
        query = "SELECT * FROM table"
        with self.assertRaises(ValueError) as e:
            SparkLLMAssistant._extract_view_name(query)
        self.assertEqual(
            str(e.exception),
            f"The provided query: '{query}' is not valid for creating a temporary view. Expected pattern: 'CREATE TEMP VIEW [VIEW_NAME] ...'",
        )

    def test_extract_view_name_with_case_insensitive_create_temp_query(self):
        """Tests if the function correctly handles case insensitivity in the CREATE TEMP VIEW keyword"""
        query = "create temp view temp_view AS SELECT * FROM table"
        expected_view_name = "temp_view"
        actual_view_name = SparkLLMAssistant._extract_view_name(query)
        self.assertEqual(actual_view_name, expected_view_name)

    def test_extract_view_name_with_empty_query(self):
        """Tests if the function correctly raises a ValueError for an empty query"""
        query = ""
        with self.assertRaises(ValueError) as e:
            SparkLLMAssistant._extract_view_name(query)
        self.assertEqual(
            str(e.exception),
            f"The provided query: '{query}' is not valid for creating a temporary view. Expected pattern: 'CREATE TEMP VIEW [VIEW_NAME] ...'",
        )


class URLTestCase(unittest.TestCase):
    def test_is_http_or_https_url_with_valid_http_url(self):
        """Tests if the function correctly identifies a valid http URL"""
        url = "http://www.example.com"
        result = SparkLLMAssistant._is_http_or_https_url(url)
        self.assertTrue(result)

    def test_is_http_or_https_url_with_valid_https_url(self):
        """Tests if the function correctly identifies a valid https URL"""
        url = "https://www.example.com"
        result = SparkLLMAssistant._is_http_or_https_url(url)
        self.assertTrue(result)

    def test_is_http_or_https_url_with_invalid_url_scheme(self):
        """Tests if the function correctly identifies an invalid URL scheme"""
        url = "ftp://www.example.com"
        result = SparkLLMAssistant._is_http_or_https_url(url)
        self.assertFalse(result)

    def test_is_http_or_https_url_with_empty_string(self):
        """Tests if the function correctly identifies an empty string as an invalid URL"""
        url = ""
        result = SparkLLMAssistant._is_http_or_https_url(url)
        self.assertFalse(result)

    def test_is_http_or_https_url_with_url_without_scheme(self):
        """Tests if the function correctly identifies a URL without a scheme as invalid"""
        url = "www.example.com"
        result = SparkLLMAssistant._is_http_or_https_url(url)
        self.assertFalse(result)

class SparkTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (SparkSession
                     .builder
                     .master("local[*]")
                     .appName("Unit-tests")
                     .getOrCreate())

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

class CacheRetrievalTestCase(SparkTestCase):
    def setUp(self):
        self.assistant = SparkLLMAssistant(cache_file_location="tests/test_cache.json")
        self.languages_df1 = self.assistant.create_df("top 5 most popular programming languages 2022")
        self.assistant.commit()

    def test_create_df(self):
        languages_df2 = self.assistant.create_df("top 5 most popular programming languages 2022")

        assert_df_equality(self.languages_df1, languages_df2)

    def test_transform_df(self):
        transform_df1 = self.assistant.transform_df(self.languages_df1, "alphabetical order by programming language")
        self.assistant.commit()
        transform_df2 = self.assistant.transform_df(self.languages_df1, "alphabetical order by programming language")

        assert_df_equality(transform_df1, transform_df2)

    def test_explain_df(self):
        explain1 = self.assistant.explain_df(self.languages_df1)
        self.assistant.commit()
        explain2 = self.assistant.explain_df(self.languages_df1)

        self.assertEqual(explain1, explain2)

    def test_udf(self):
        @self.assistant.udf
        def udf1(s: str) -> str:
            """reverse letters in string s"""

        self.assistant.commit()

        @self.assistant.udf
        def udf2(s: str) -> str:
            """reverse letters in string s"""

        self.assertEqual(udf1("test"), udf2("test"))


if __name__ == "__main__":
    unittest.main()
