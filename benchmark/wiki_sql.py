import json
import re
from argparse import ArgumentParser

from pyspark.sql import SparkSession

from pyspark_ai import SparkAI


def generate_sql_statements(table_file):
    sql_statements = []

    with open(table_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())

            table_name = item['id']
            # quote the headers with backticks
            headers = ["`{}`".format(h) for h in item['header']]
            header_str = "(" + ",".join(headers) + ")"
            rows = item['rows']

            values_str_list = []

            for row in rows:
                # Convert each value in the row to a string and escape single quotes
                values_str_list.append("(" + ",".join(["'{}'".format(str(val).replace("'", "''")) for val in row]) + ")")

            values_str = ",".join(values_str_list)
            create_statement = f"CREATE TEMP VIEW `{table_name}` AS SELECT * FROM VALUES {values_str} as {header_str};"
            sql_statements.append(create_statement)

    return sql_statements


# Parse questions and tables from the source file
def get_tables_and_questions(source_file):
    tables = []
    questions = []
    with open(source_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            tables.append(item['table_id'])
            questions.append(item['question'])
    return tables, questions


def convert_to_wikisql_format(sql_query, table_schema):
    # Predefined lists for lookup
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']

    # Find the selected column and aggregation operation (if any)
    select_pattern = re.compile(r"SELECT\s+(?:([A-Z]+)\s*\()?(?:`?([^`]+)`?)?\)?")
    select_match = select_pattern.search(sql_query)
    agg_operation, sel_col = select_match.groups()

    # If there's an aggregation operation, get its index
    agg_index = 0 if not agg_operation else agg_ops.index(agg_operation)

    # Get the index of the selected column from the table schema
    sel_col_index = table_schema.index(sel_col) + 1 if sel_col in table_schema else 0

    # Extract the condition column, its operation, and value (if present)
    where_pattern = re.compile(r"WHERE\s+`?([^`]+)`?\s+([=><])\s+(?:'([^']+)'|(\d+\.?\d*))")
    where_match = where_pattern.search(sql_query)

    conds = []
    if where_match:
        cond_col, cond_op, string_val, num_val = where_match.groups()
        # If string value is present, use it. If num_val is an integer, convert to int, else float.
        if string_val:
            cond_value = string_val
        elif "." in num_val:
            cond_value = float(num_val)
        else:
            cond_value = int(num_val)

        # Get the index of the condition column from the table schema
        cond_col_index = table_schema.index(cond_col) + 1
        cond_op_index = cond_ops.index(cond_op)

        conds.append([cond_col_index, cond_op_index, cond_value])

    # Compile into desired format
    result = {
        "query": {
            "sel": sel_col_index,
            "agg": agg_index,
            "conds": conds
        },
        "error": ""
    }

    return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('table_file', help='table definition file')
    parser.add_argument('source_file', help='source file for the prediction')
    args = parser.parse_args()

    # Example usage:
    table_file = args.table_file
    statements = generate_sql_statements(table_file)
    spark = SparkSession.builder.getOrCreate()
    for stmt in statements:
        spark.sql(stmt)

    source_file = args.source_file
    tables, questions = get_tables_and_questions(source_file)
    spark_ai = SparkAI(spark_session=spark)
    # Create sql query for each question and table
    for table, question in zip(tables, questions):
        df = spark.table(f"`{table}`")
        sql_query = spark_ai._get_transform_sql_query(df, question, cache=False)
        print(sql_query)

