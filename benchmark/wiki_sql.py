import json
import re
from argparse import ArgumentParser

from babel.numbers import parse_decimal, NumberFormatError
from pyspark.sql import SparkSession

from pyspark_ai import SparkAI


def generate_sql_statements(table_file):
    sql_statements = []
    num_re = re.compile(r'[-+]?\d*\.\d+|\d+')
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
                vals = []
                for val in row:
                    if isinstance(val, str):
                        val = "'{}'".format(val.lower().replace("'", "''").replace("\\", "\\\\"))
                    else:
                        val = str(float(val))
                    vals.append(val)

                # Convert each value in the row to a string and escape single quotes
                values_str_list.append("(" + ",".join(vals) + ")")

            values_str = ",".join(values_str_list)
            create_statement = f"CREATE TEMP VIEW `{table_name}` AS SELECT * FROM VALUES {values_str} as {header_str};"
            sql_statements.append(create_statement)

    return sql_statements


# Parse questions and tables from the source file
def get_tables_and_questions(source_file):
    tables = []
    questions = []
    results = []
    with open(source_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            tables.append(item['table_id'])
            questions.append(item['question'])
            results.append(item['result'])
    return tables, questions, results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('table_file', help='table definition file')
    parser.add_argument('source_file', help='source file for the prediction')
    args = parser.parse_args()

    # Example usage:
    table_file = args.table_file
    statements = generate_sql_statements(table_file)
    spark = SparkSession.builder.getOrCreate()
    # spark.conf.set("spark.sql.caseSensitive", "true")
    for stmt in statements:
        spark.sql(stmt)

    source_file = args.source_file
    tables, questions, results = get_tables_and_questions(source_file)
    spark_ai = SparkAI(spark_session=spark)
    matched = 0
    # Create sql query for each question and table
    with open("pyspark_ai.jsonl", "w") as file:
        for table, question, result in zip(tables, questions, results):
            print(question)
            df = spark.table(f"`{table}`")
            try:
                query = spark_ai._get_transform_sql_query(df=df, desc=question, cache=True)
                result_df = spark.sql(query.lower())
            except Exception as e:
                print(e)
                continue
            spark_ai.commit()
            for i in range(len(result_df.columns)):
                answer = result_df.rdd.map(lambda row: row[i]).collect()
                if answer == result:
                    matched += 1
                    break
                else:
                    print(f"Answer: {answer} does not match {result}")

    print(f"Matched {matched} out of {len(results)}")

