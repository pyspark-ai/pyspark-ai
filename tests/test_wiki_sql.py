import unittest

from benchmark.wiki_sql import convert_to_wikisql_format


class WikiSQLTestCase(unittest.TestCase):
    def test_convert_to_wikisql_format(self):
        table_schema = ["No. in set", "Country ( exonym )", "col3", "col4", "col5", "col6"]
        sql_query1 = "SELECT `Country ( exonym )` AS result FROM table_1_10015132_11 WHERE `No. in set` = '3'"
        print(convert_to_wikisql_format(sql_query1, table_schema))

        sql_query2 = "SELECT COUNT(col5) AS result FROM table_1_10015132_11 WHERE `col6` = 1.234"
        print(convert_to_wikisql_format(sql_query2, table_schema))
