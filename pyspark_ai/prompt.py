# flake8: noqa
from langchain import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

SEARCH_TEMPLATE = """Given a Query and a list of Google Search Results, return the link
from a reputable website which contains the data set to answer the question. {columns}
Query:{query}
Google Search Results:
```
{search_results}
```
The answer MUST contain the url link only
"""

SEARCH_PROMPT = PromptTemplate(
    input_variables=["query", "search_results", "columns"], template=SEARCH_TEMPLATE
)

SQL_TEMPLATE = """Given the following question:
```
{query}
```
I got the following answer from a web page:
```
{web_content}
```
Now help me write a SQL query to store the answer into a temp view.
Give each column a clearly descriptive name (no abbreviations).
If a column can be either String or Numeric, ingest it as Numeric.
Here is an example of how to store data into the temp view {view_name}:
```
CREATE OR REPLACE TEMP VIEW {view_name} AS SELECT * FROM VALUES('Citizen Kane', 1941), ('Schindler\'s List', 1993) AS v1(title, year)
```
{columns}
The answer MUST contain query only and the temp view MUST be {view_name}.
"""

SQL_PROMPT = PromptTemplate(
    input_variables=["query", "web_content", "view_name", "columns"],
    template=SQL_TEMPLATE,
)


SPARK_SQL_EXAMPLES = [
    """QUESTION: Given a Spark temp view `spark_ai_temp_view_93bcf0` with the following columns:
```
Product STRING
Amount BIGINT
Country STRING
```
Write a Spark SQL query to retrieve from view `spark_ai_temp_view_93bcf0`: Pivot the fruit table by country and sum the amount for each fruit and country combination.
Thought: Spark SQL does not support dynamic pivot operations, which are required to transpose the table as requested. I should get all the distinct values of column country.
Action: query_sql_db
Action Input: "SELECT DISTINCT Country FROM spark_ai_temp_view_93bcf0"
Observation: USA, Canada, Mexico, China
Thought: I can write a query to pivot the table by country and sum the amount for each fruit and country combination.
Action: query_validation
Action Input: SELECT * FROM spark_ai_temp_view_93bcf0 PIVOT (SUM(Amount) FOR Country IN ('USA', 'Canada', 'Mexico', 'China'))
Observation: OK
Thought:I now know the final answer.
Final Answer: SELECT * FROM spark_ai_temp_view_93bcf0 PIVOT (SUM(Amount) FOR Country IN ('USA', 'Canada', 'Mexico', 'China'))"""
    """QUESTION: Given a Spark temp view `spark_ai_temp_view_19acs2` with the following columns:
```
Car STRING
Color STRING
Make STRING
```
Write a Spark SQL query to retrieve from view `spark_ai_temp_view_19acs2`: If color is gold, what is the total number of cars?
Thought: I will use the column 'Color' to filter the rows where its value is 'gold' and then select the COUNT(`Car`) 
because COUNT gives me the total number of cars.
Action: query_validation
Action Input: SELECT COUNT(`Car`) FROM spark_ai_temp_view_19acs2 WHERE `Color` = 'gold'
Observation: OK
Thought: I now know the final answer.
Final Answer: SELECT COUNT(`Car`) FROM spark_ai_temp_view_19acs2 WHERE `Color` = 'gold'"""
    """QUESTION: Given a Spark temp view `spark_ai_temp_view_wl2sdf` with the following columns:
```
PassengerId INT
Survived INT
Pclass INT
Name STRING
Sex STRING
Age DOUBLE
SibSp INT
Parch INT
Ticket STRING
Fare DOUBLE
Cabin STRING
Embarked STRING
```
Write a Spark SQL query to retrieve from view `spark_ai_temp_view_wl2sdf`: What's the name of the oldest survived passenger?
Thought: I will query the Name and Age columns, filtering by Survived and ordering by Age in descending order.
Action: query_validation
Action Input: SELECT Name, Age FROM spark_ai_temp_view_wl2sdf WHERE Survived = 1 ORDER BY Age DESC LIMIT 1
Observation: OK
Thought:I now know the final answer.
Final Answer: SELECT Name, Age FROM spark_ai_temp_view_wl2sdf WHERE Survived = 1 ORDER BY Age DESC LIMIT 1"""
]

SPARK_SQL_SUFFIX = """\nQuestion: Given a Spark temp view `{view_name}` {comment}.
The dataframe contains the column names and types in this format:
column_name: type.
It's very important to ONLY use the verbatim column names in your resulting SQL query.
For example, the verbatim column name from the line 'Fruit Color: string' is `Fruit Color`.
So, a resulting SQL query would be: SELECT * FROM view WHERE `Fruit Color` = 'orange'.
The verbatim column name from the line 'Number of years: int' is `Number of years`.
So, a resulting SQL query would be: SELECT * FROM view WHERE `Number of years` = 3.

Here are the column names and types for your dataframe:
```
{columns}
```

Here are some sample rows from your dataframe:
{sample_rows}

Write a Spark SQL query to retrieve the following from view `{view_name}`: {desc}
{agent_scratchpad}"""

SPARK_SQL_PREFIX = """You are an assistant for writing professional Spark SQL queries. 
Given a question, you need to write a Spark SQL query to answer the question. The result is ALWAYS a Spark SQL query.
"""

SPARK_SQL_PROMPT = PromptTemplate.from_examples(
    examples=SPARK_SQL_EXAMPLES,
    suffix=SPARK_SQL_SUFFIX,
    input_variables=[
        "view_name",
        "columns",
        "sample_rows",
        "comment",
        "desc",
        "agent_scratchpad",
    ],
    prefix=SPARK_SQL_PREFIX,
)


EXPLAIN_PREFIX = """You are an Apache Spark SQL expert, who can summary what a dataframe retrieves. Given an analyzed
query plan of a dataframe, you will
1. convert the dataframe to SQL query. Note that an explain output contains plan
nodes separated by `\\n`. Each plan node has its own expressions and expression ids.
2. summary what the sql query retrieves.
"""

EXPLAIN_SUFFIX = "analyzed_plan: {input}\nexplain:"

_plan1 = """
GlobalLimit 100
    +- LocalLimit 100
       +- Sort [d_year ASC NULLS FIRST, sum_agg DESC NULLS LAST, brand_id ASC NULLS FIRST], true
          +- Aggregate [d_year, i_brand, i_brand_id], [d_year, i_brand_id AS brand_id, i_brand AS brand, sum(ss_ext_sales_price) AS sum_agg]
             +- Filter (((d_date_sk = ss_sold_date_sk) AND (ss_item_sk = i_item_sk)) AND ((i_manufact_id = 128) AND (d_moy = 11)))
                +- Join Inner
                   :- Join Inner
                   :  :- SubqueryAlias dt
                   :  :  +- SubqueryAlias spark_catalog.tpcds_sf1_delta.date_dim
                   :  :     +- Relation spark_catalog.tpcds_sf1_delta.date_dim[d_date_sk,d_date_id,d_date,d_month_seq,d_week_seq,d_quarter_seq,d_year,d_dow,d_moy,d_dom,d_qoy,d_fy_year,d_fy_quarter_seq,d_fy_week_seq,d_day_name,d_quarter_name,d_holiday,d_weekend,d_following_holiday,d_first_dom,d_last_dom,d_same_day_ly,d_same_day_lq,d_current_day,... 4 more fields] parquet
                   :  +- SubqueryAlias spark_catalog.tpcds_sf1_delta.store_sales
                   :     +- Relation spark_catalog.tpcds_sf1_delta.store_sales[ss_sold_date_sk,ss_sold_time_sk,ss_item_sk,ss_customer_sk,ss_cdemo_sk,ss_hdemo_sk,ss_addr_sk,ss_store_sk,ss_promo_sk,ss_ticket_numberL,ss_quantity,ss_wholesale_cost,ss_list_price,ss_sales_price,ss_ext_discount_amt,ss_ext_sales_price,ss_ext_wholesale_cost,ss_ext_list_price,ss_ext_tax,ss_coupon_amt,ss_net_paid,ss_net_paid_inc_tax,ss_net_profit] parquet
                   +- SubqueryAlias spark_catalog.tpcds_sf1_delta.item
                      +- Relation spark_catalog.tpcds_sf1_delta.item[i_item_sk,i_item_id,i_rec_start_date,i_rec_end_date,i_item_desc,i_current_price,i_wholesale_cost,i_brand_id,i_brand,i_class_id,i_class,i_category_id,i_category,i_manufact_id,i_manufact,i_size,i_formulation,i_color,i_units,i_container,i_manager_id,i_product_name] parquet
"""

_explain1 = """
The analyzed plan can be translated into the following SQL query:
```sql
SELECT
  dt.d_year,
  item.i_brand_id brand_id,
  item.i_brand brand,
  SUM(ss_ext_sales_price) sum_agg
FROM date_dim dt, store_sales, item
WHERE dt.d_date_sk = store_sales.ss_sold_date_sk
  AND store_sales.ss_item_sk = item.i_item_sk
  AND item.i_manufact_id = 128
  AND dt.d_moy = 11
GROUP BY dt.d_year, item.i_brand, item.i_brand_id
ORDER BY dt.d_year, sum_agg DESC, brand_id
LIMIT 100
```
In summary, this dataframe is retrieving the top 100 brands (specifically of items manufactured by manufacturer with id 128) with the highest total sales price for each year in the month of November. It presents the results sorted by year, total sales (in descending order), and brand id.
"""

_explain_examples = [{"analyzed_plan": _plan1, "explain": _explain1}]

_example_formatter = """
analyzed_plan: {analyzed_plan}
explain: {explain}
"""

_example_prompt = PromptTemplate(
    input_variables=["analyzed_plan", "explain"], template=_example_formatter
)

EXPLAIN_DF_PROMPT = FewShotPromptTemplate(
    examples=_explain_examples,
    example_prompt=_example_prompt,
    prefix=EXPLAIN_PREFIX,
    suffix=EXPLAIN_SUFFIX,
    input_variables=["input"],
    example_separator="\n\n",
)

PLOT_PROMPT_TEMPLATE = """
You are an Apache Spark SQL expert programmer.
It is forbidden to include old deprecated APIs in your code.
For example, you will not use the pandas method "append" because it is deprecated.

Given a pyspark DataFrame `df`, with the output columns:
{columns}

And an explanation of `df`: {explain}

Write Python code to visualize the result of `df` using plotly. Do any aggregation against `df` 
first, before converting the `df` to a pandas DataFrame. Make sure to use the exact column names 
of `df`.
Your code may NOT contain "append" anywhere. Instead of append, use pd.concat.
There is no need to install any package with pip. Do include any necessary import statements.
Display the plot directly, instead of saving into an HTML.
Do not use scatter plot to display any kind of percentage data.
You must import and start your Spark session if you use a Spark DataFrame.
Remember to ensure that your code does NOT include "append" anywhere, under any circumstance (use pd.concat instead).

Ensure that your code is correct.
{instruction}
"""

PLOT_PROMPT = PromptTemplate(
    input_variables=["columns", "explain", "instruction"], template=PLOT_PROMPT_TEMPLATE
)

VERIFY_TEMPLATE = """
Given 1) a PySpark dataframe, df, and 2) a description of expected properties, desc,
generate a Python function to test whether the given dataframe satisfies the expected properties.
Your generated function should take 1 parameter, df, and the return type should be a boolean.
You will call the function, passing in df as the parameter, and return the output (True/False).

In total, your output must follow the format below, exactly (no explanation words):
1. function definition f, in Python (Do NOT surround the function definition with quotes)
2. 1 blank new line
3. Call f on df and assign the result to a variable, result: result = name_of_f(df)
The answer MUST contain python code only. For example, do NOT include "Here is your output:"

Include any necessary import statements INSIDE the function definition, like this:
def gen_random():
    import random
    return random.randint(0, 10)

Your output must follow the format of the example below, exactly:
Input:
df = DataFrame[name: string, age: int]
desc = "expect 5 columns"

Output:
def has_5_columns(df) -> bool:
    # Get the number of columns in the DataFrame
    num_columns = len(df.columns)

    # Check if the number of columns is equal to 5
    if num_columns == 5:
        return True
    else:
        return False

result = has_5_columns(df)

No explanation words (e.g. do not say anything like "Here is your output:")

Here is your input df: {df}
Here is your input description: {desc}
"""

VERIFY_PROMPT = PromptTemplate(input_variables=["df", "desc"], template=VERIFY_TEMPLATE)

UDF_PREFIX = """
This is the documentation for a PySpark user-defined function (udf): pyspark.sql.functions.udf

A udf creates a deterministic, reusable function in Spark. It can take any data type as a parameter,
and by default returns a String (although it can return any data type).
The point is to reuse a function on several dataframes and SQL functions.

Given 1) input arguments, 2) a description of the udf functionality,
3) the udf return type, and 4) the udf function name,
generate and return a callable udf.

Return ONLY the callable resulting udf function (no explanation words).
Include any necessary import statements INSIDE the function definition.
For example:
def gen_random():
    import random
    return random.randint(0, 10)
"""

UDF_SUFFIX = """
input_args_types: {input_args_types}
input_desc: {desc}
return_type: {return_type}
udf_name: {udf_name}
output:\n
"""

_udf_output1 = """
def to_upper(s) -> str:
    if s is not None:
        return s.upper()
"""

_udf_output2 = """
def add_one(x) -> int:
    if x is not None:
        return x + 1
"""

_udf_examples = [
    {
        "input_args_types": "(s: str)",
        "desc": "Convert string s to uppercase",
        "return_type": "str",
        "udf_name": "to_upper",
        "output": _udf_output1,
    },
    {
        "input_args_types": "(x: int)",
        "desc": "Add 1",
        "return_type": "int",
        "udf_name": "add_one",
        "output": _udf_output2,
    },
]

_udf_formatter = """
input_args_types: {input_args_types}
desc: {desc}
return_type: {return_type}
udf_name: {udf_name}
output: {output}
"""

_udf_prompt = PromptTemplate(
    input_variables=["input_args_types", "desc", "return_type", "udf_name", "output"],
    template=_udf_formatter,
)

UDF_PROMPT = FewShotPromptTemplate(
    examples=_udf_examples,
    example_prompt=_udf_prompt,
    prefix=UDF_PREFIX,
    suffix=UDF_SUFFIX,
    input_variables=["input_args_types", "desc", "return_type", "udf_name"],
    example_separator="\n\n",
)
