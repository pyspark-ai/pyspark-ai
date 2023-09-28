import json
import re
from argparse import ArgumentParser

from babel.numbers import parse_decimal, NumberFormatError
from pyspark.sql import SparkSession

from pyspark_ai import SparkAI


def replace_quotes_and_backslashes(s):
    return s.replace("'", "''").replace("\\", "\\\\")


# Generate ingestion SQL statements from the table definition file, using `CREATE TEMP VIEW ... AS SELECT`.
def create_temp_view_statements(table_file):
    sql_statements = []
    with open(table_file, "r") as f:
        for line in f:
            item = json.loads(line.strip())

            table_name = get_table_name(item["id"])
            # quote the headers with backticks
            headers = ["`{}`".format(h) for h in item["header"]]
            header_str = "(" + ",".join(headers) + ")"
            rows = item["rows"]

            values_str_list = []

            for row in rows:
                vals = []
                for val in row:
                    if isinstance(val, str):
                        val = "'{}'".format(replace_quotes_and_backslashes(val.lower()))
                    else:
                        val = str(float(val))
                    vals.append(val)

                # Convert each value in the row to a string and escape single quotes
                values_str_list.append("(" + ",".join(vals) + ")")

            values_str = ",".join(values_str_list)
            # if key section title exist, add it to the comment
            if "section_title" in item:
                section_title = item["section_title"] + " of "
            else:
                section_title = ""

            if "page_title" in item:
                page_title = item["page_title"]
            else:
                page_title = ""
            comment = section_title + page_title
            create_statement = f'CREATE TABLE IF NOT EXISTS `{table_name}` USING ORC comment "{comment}" ' \
                               f'AS SELECT * FROM VALUES {values_str} as {header_str};'
            sql_statements.append(create_statement)

    return sql_statements


def get_table_name(table_id: str) -> str:
    # map table id like '1-1004033-1' to 'table_1_1004033_1'
    return "table_" + table_id.replace("-", "_")


# Reconstruction of the original query from the table id and standard query format
def get_sql_query(table_id, select_index, aggregation_index, conditions):
    agg_ops = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
    cond_ops = ["=", ">", "<", "OP"]
    num_re = re.compile(r"[-+]?\d*\.\d+|\d+")
    df = spark.table(f"`{get_table_name(table_id)}`")
    select = df.columns[select_index]
    agg = agg_ops[aggregation_index]
    if agg != 0:
        select = f"{agg}({select})"
    where_clause = []
    for col_index, op, val in conditions:
        if isinstance(val, str):
            val = f"'{val.lower()}'"
        if df.dtypes[col_index][1] == "double" and not isinstance(val, (int, float)):
            try:
                val = float(parse_decimal(val))
            except NumberFormatError:
                val = float(num_re.findall(val)[0])
        where_clause.append(f"`{df.columns[col_index]}` {cond_ops[op]} {val}")
    where_str = ""
    if where_clause:
        where_str = "WHERE " + " AND ".join(where_clause)
    return f"SELECT `{select}` FROM `{table_id}` {where_str}"


# Parse questions and tables from the source file
def get_tables_and_questions(source_file):
    tables = []
    questions = []
    results = []
    sqls = []
    with open(source_file, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            tables.append(item["table_id"])
            questions.append(item["question"])
            results.append(item["result"])
            sqls.append(item["sql"])
    return tables, questions, results, sqls


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--table_file",
        help="table definition file",
        default="data/test_sample.tables.jsonl",
    )
    parser.add_argument(
        "--source_file",
        help="source file for the prediction",
        default="data/test_sample.jsonl",
    )
    args = parser.parse_args()

    table_file = args.table_file
    statements = create_temp_view_statements(table_file)
    spark = SparkSession.builder.getOrCreate()
    for stmt in statements:
        spark.sql(stmt)

    source_file = args.source_file
    tables, questions, results, sqls = get_tables_and_questions(source_file)
    spark_ai = SparkAI(spark_session=spark, vector_store_dir="data", verbose=False)
    matched = 0
    not_matched = 0
    errors = 0

    # Create sql query for each question and table
    for table, question, expected_result, sql in zip(tables, questions, results, sqls):
        try:
            df = spark.table(f"`{get_table_name(table)}`")
        except Exception as e:
            errors += 1
            print(e)
            continue
        try:
            query = spark_ai._get_transform_sql_query(
                df=df, desc=question, cache=False
            ).lower()
            result_df = spark.sql(query)
        except Exception as e:
            errors += 1
            print(e)
            continue
        spark_ai.commit()
        found_match = False
        spark_ai_result = []

        for i in range(len(result_df.columns)):
            spark_ai_result = result_df.rdd.map(lambda row: row[i]).collect()

            spark_ai_result = [str(ele) for ele in spark_ai_result]
            expected_result = [str(ele) for ele in expected_result]

            # sort spark_ai_result and expected_result, to account for unpredictable row order
            spark_ai_result = sorted(spark_ai_result)
            expected_result = sorted(expected_result)

            actual_phrase = " ".join(spark_ai_result)
            expected_phrase = " ".join(expected_result)

            if actual_phrase == expected_phrase:
                matched += 1
                found_match = True
                break

        if not found_match:
            not_matched += 1
            print("Question: {}".format(question))
            print(
                "Expected query: {}".format(
                    get_sql_query(table, sql["sel"], sql["agg"], sql["conds"])
                )
            )
            print("Actual query: {}".format(query))
            print("Expected result: {}".format(expected_result))
            print("Actual result: {}".format(spark_ai_result))
            print("")

    print(f"Matched: {matched} out of {len(results)}")
    print(f"Incorrect: {not_matched} out of {len(results)}")
    print(f"Errors: {errors} out of {len(results)}")
