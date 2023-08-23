from benchmark.wiki_sql import convert_to_wikisql_format

import unittest

class TestWikiSQLConversion(unittest.TestCase):

    def setUp(self):
        self.table_schema = ["No. in set", "Country ( exonym )", "col3", "col4", "col5", "col6"]

    def test_simple_select(self):
        sql_query = "SELECT `Country ( exonym )` AS result FROM table_1_10015132_11 WHERE `No. in set` = '3'"
        expected = {"query": {"sel": 1, "conds": [[0, 0, '3']], "agg": 0}, "error": ""}
        self.assertEqual(convert_to_wikisql_format(sql_query, self.table_schema), expected)

    def test_count_aggregation(self):
        sql_query = "SELECT COUNT(`col5`) AS result FROM table_1_10015132_11 WHERE `No. in set` = '3'"
        expected = {"query": {"sel": 4, "conds": [[0, 0, '3']], "agg": 3}, "error": ""}
        self.assertEqual(convert_to_wikisql_format(sql_query, self.table_schema), expected)

    def test_max_aggregation(self):
        sql_query = "SELECT MAX(col5) AS result FROM table_1_10015132_11 WHERE `No. in set` = '3'"
        expected = {"query": {"sel": 4, "conds": [[0, 0, '3']], "agg": 1}, "error": ""}
        self.assertEqual(convert_to_wikisql_format(sql_query, self.table_schema), expected)

    def test_float_condition(self):
        sql_query = "SELECT `col3` FROM table_1_10015132_11 WHERE `col4` = 12.34"
        expected = {"query": {"sel": 2, "conds": [[3, 0, 12.34]], "agg": 0}, "error": ""}
        self.assertEqual(convert_to_wikisql_format(sql_query, self.table_schema), expected)

    def test_integer_condition(self):
        sql_query = "SELECT `col3` FROM table_1_10015132_11 WHERE `col4` = 12"
        expected = {"query": {"sel": 2, "conds": [[3, 0, 12]], "agg": 0}, "error": ""}
        self.assertEqual(convert_to_wikisql_format(sql_query, self.table_schema), expected)

    def test_no_conditions(self):
        sql_query = "SELECT `Country ( exonym )` AS result FROM table_1_10015132_11"
        expected = {"query": {"sel": 1, "conds": [], "agg": 0}, "error": ""}
        self.assertEqual(convert_to_wikisql_format(sql_query, self.table_schema), expected)

    def test_multiple_select_columns(self):
        sql_query = "SELECT `Country ( exonym )`, `No. in set` FROM table_1_10015132_11 WHERE `col3` = 'value'"
        expected = {"query": {"sel": 1, "conds": [[2, 0, 'value']], "agg": 0}, "error": ""}
        self.assertEqual(convert_to_wikisql_format(sql_query, self.table_schema), expected)

    def test_distinct_count_aggregation(self):
        sql_query = "SELECT COUNT(DISTINCT `col5`) FROM table_1_10015132_11 WHERE `col4` = 12"
        expected = {"query": {"sel": 4, "conds": [[3, 0, 12]], "agg": 3}, "error": ""}
        self.assertEqual(convert_to_wikisql_format(sql_query, self.table_schema), expected)

    def test_distinct_avg_aggregation(self):
        sql_query = "SELECT AVG(DISTINCT `Country ( exonym )`) FROM table_1_10015132_11 WHERE `No. in set` = '5'"
        expected = {"query": {"sel": 1, "conds": [[0, 0, '5']], "agg": 5}, "error": ""}
        self.assertEqual(convert_to_wikisql_format(sql_query, self.table_schema), expected)


if __name__ == '__main__':
    unittest.main()

