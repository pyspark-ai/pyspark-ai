import contextlib
import io
from pyfakefs.fake_filesystem_unittest import Patcher

from pyspark.sql import SparkSession
from pyspark_ai import SparkAI
from pyspark_ai.ai_utils import AIUtils
from benchmark.plot.benchmark_util import *


def prep_df(spark_ai):
    data = [
        ("2023-08-30", 184.94, 187.85, 184.74, 187.65, 60590000),
        ("2023-08-31", 187.70, 190.00, 187.50, 189.50, 58500000),
        ("2023-09-01", 189.60, 192.00, 189.40, 191.00, 57400000),
        ("2023-09-02", 191.10, 193.50, 190.90, 192.50, 56300000),
        ("2023-09-03", 192.60, 195.00, 192.40, 194.00, 55200000),
        ("2023-09-04", 194.10, 196.50, 193.90, 195.50, 54100000),
        ("2023-09-05", 195.60, 198.00, 195.40, 197.00, 53000000),
        ("2023-09-06", 184.94, 187.85, 184.74, 187.65, 60540000),
        ("2023-09-07", 188.70, 191.00, 187.50, 189.50, 58800000),
        ("2023-09-08", 189.60, 192.00, 187.40, 191.50, 57300000),
        ("2023-09-09", 191.10, 193.50, 190.90, 192.50, 56900000),
        ("2023-09-10", 192.60, 196.00, 192.40, 194.00, 54200000),
        ("2023-09-11", 194.10, 196.50, 193.90, 194.50, 54800000),
        ("2023-09-12", 195.60, 198.00, 195.40, 196.00, 53600000),
    ]
    df = spark_ai._spark.createDataFrame(data, ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])
    return df

import logging
from io import StringIO

def capture_plot_logs(df, desc_plot, spark_ai):
    root_logger = logging.getLogger()
    buffer = StringIO()
    ch = logging.StreamHandler(buffer)
    root_logger.addHandler(ch)

    with spark_ai._logger.disable_code_colorization():
        df.ai.plot(desc_plot)

    log_contents = buffer.getvalue()
    root_logger.removeHandler(ch)
    return log_contents


def substitute_show_to_json(string_list):
    import re
    modified_list = []
    for string in string_list:
        modified_string = re.sub(r'(\w+)\.show\(\)', r'print(\1.to_json())', string)
        modified_list.append(modified_string)
    return modified_list


def dict_diff(dict1, dict2):
    diff = {}

    # Check keys in dict1 but not in dict2
    for key in dict1.keys():
        if key not in dict2:
            diff[key] = ("Only in dict1:", dict1[key])
        elif dict1[key] != dict2[key]:
            diff[key] = ("Different values:", dict1[key], dict2[key])

    # Check keys in dict2 but not in dict1
    for key in dict2.keys():
        if key not in dict1:
            diff[key] = ("Only in dict2:", dict2[key])

    return diff


def generate_golden_json(df):
    fig_dicts = []
    fig_dicts.append(f1(df))
    fig_dicts.append(f2(df))
    fig_dicts.append(f3(df))
    fig_dicts.append(f4(df))
    fig_dicts.append(f5(df))
    fig_dicts.append(f6(df))
    fig_dicts.append(f7(df))
    fig_dicts.append(f8(df))
    fig_dicts.append(f9(df))
    fig_dicts.append(f10(df))

    # Use the fake filesystem only for this operation
    with open("output.json", "w") as f:
        json.dump(fig_dicts, f)


def main():
    """
    Performs benchmarking by capturing plot logs, substituting ".show()" with ".to_json()",
    and comparing the results with golden data.
    """
    spark = SparkSession.builder.getOrCreate()
    spark_ai = SparkAI(spark_session=spark, verbose=True)
    spark_ai.activate()
    df = prep_df(spark_ai)


    desc_plots = [
        "bar plot of daily closing prices",
        "histogram of daily price changes",
        "box plot of daily closing prices",
        "line plot of daily opening prices",
        "scatter plot with opening prices on the x-axis and closing prices on the y-axis",
        "pie chart comparing days with opening prices above 190.00",
        "area chart of daily closing prices",
        "bar plot of daily high-low range",
        "histogram of daily trading volume",
        "line plot of daily trading volume change",
    ]

    file_path = "output.json"

    with Patcher() as patcher:  # Use the fake filesystem only for this operation
        generate_golden_json(df)
        with open(file_path, 'r') as file:
            content = file.read()

        # Remove the file from the fake filesystem after reading its content
        if patcher.fs.exists(file_path):  # Check if the file exists in the fake filesystem
            patcher.fs.remove(file_path)

    goldens = json.loads(content)

    # Benchmark
    err_cnt = 0

    for i in range(len(desc_plots)):
        captured_output = capture_plot_logs(df, desc_plots[i], spark_ai)
        codeblocks = AIUtils.extract_code_blocks(captured_output)
        sub_codeblocks = substitute_show_to_json(codeblocks)
        code = "\n".join(sub_codeblocks)

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile(code, "plot_df-CodeGen-benchmark", "exec"))
        captured_output = buffer.getvalue()[:-1]
        predicted = json.loads(captured_output)

        if not all(predicted['data'][0].get(key) == goldens[i]['data'][0].get(key) for key in \
                goldens[i]['data'][0]):
            print("[ERROR] " + desc_plots[i])
            print("[PREDICTED]")
            print(predicted['data'])
            print("[GOLDEN]")
            print(goldens[i]['data'])
            err_cnt += 1

        buffer.close()

    print(f"{err_cnt} out of {len(desc_plots)} failed")


if __name__ == '__main__':
    main()
