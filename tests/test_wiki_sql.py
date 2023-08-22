from benchmark.wiki_sql import convert_to_wikisql_format

import unittest

class TestWikiSQLConversion(unittest.TestCase):

    def setUp(self):
        self.table_schema = ["No. in set", "Country ( exonym )", "col3", "col4", "col5", "col6"]

    def test_simple_select(self):
        sql_query = "SELECT `Country ( exonym )` AS result FROM table_1_10015132_11 WHERE `No. in set` = '3'"
        expected = {"query": {"sel": 2, "conds": [[1, 0, '3']], "agg": 0}, "error": ""}
        self.assertEqual(convert_to_wikisql_format(sql_query, self.table_schema), expected)

    def test_count_aggregation(self):
        sql_query = "SELECT COUNT(`col5`) AS result FROM table_1_10015132_11 WHERE `No. in set` = '3'"
        expected = {"query": {"sel": 5, "conds": [[1, 0, '3']], "agg": 3}, "error": ""}
        self.assertEqual(convert_to_wikisql_format(sql_query, self.table_schema), expected)

    def test_float_condition(self):
        sql_query = "SELECT `col3` FROM table_1_10015132_11 WHERE `col4` = 12.34"
        expected = {"query": {"sel": 3, "conds": [[4, 0, 12.34]], "agg": 0}, "error": ""}
        self.assertEqual(convert_to_wikisql_format(sql_query, self.table_schema), expected)

    def test_integer_condition(self):
        sql_query = "SELECT `col3` FROM table_1_10015132_11 WHERE `col4` = 12"
        expected = {"query": {"sel": 3, "conds": [[4, 0, 12]], "agg": 0}, "error": ""}
        self.assertEqual(convert_to_wikisql_format(sql_query, self.table_schema), expected)

    def test_no_conditions(self):
        sql_query = "SELECT `Country ( exonym )` AS result FROM table_1_10015132_11"
        expected = {"query": {"sel": 2, "conds": [], "agg": 0}, "error": ""}
        self.assertEqual(convert_to_wikisql_format(sql_query, self.table_schema), expected)

    # You can add more test cases as needed to cover other edge cases or scenarios

if __name__ == '__main__':
    unittest.main()

