import json
from argparse import ArgumentParser

from pyspark.sql import SparkSession


def generate_sql_statements(data_file):
    sql_statements = []

    with open(data_file, 'r') as f:
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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('table_file', help='table definition file')
    # parser.add_argument('source_file', help='source file for the prediction')
    args = parser.parse_args()

    # Example usage:
    data_file = args.table_file
    statements = generate_sql_statements(data_file)
    spark = SparkSession.builder.getOrCreate()
    for stmt in statements:
        print(stmt)
        spark.sql(stmt)
