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
Here is an example of how to store data into a temp view:
```
CREATE OR REPLACE TEMP VIEW movies AS SELECT * FROM VALUES(('Citizen Kane', 1941), ('Schindler\'s List', 1993)) AS v1(title, year)
```
{columns}
The answer MUST contain query only.
"""

SQL_PROMPT = PromptTemplate(
    input_variables=["query", "web_content", "columns"], template=SQL_TEMPLATE
)

TRANSFORM_TEMPLATE = """
Given a Spark temp view `{view_name}` with the following columns:
```
{columns}
```
Write a Spark SQL query to retrieve: {desc}
The answer MUST contain query only.
"""

TRANSFORM_PROMPT = PromptTemplate(
    input_variables=["view_name", "columns", "desc"], template=TRANSFORM_TEMPLATE
)

EXPLAIN_PREFIX = """You are an Apache Spark SQL expert, who can summary what a dataframe retrieves. Given an analyzed 
query plan of a dataframe, you will 
1. convert the dataframe to SQL query. Note that an explain output contains plan 
nodes separated by "\n". Each plan node has its own expressions and expression ids. 
2. summary what the sql query retrieves. 
"""

EXPLAIN_SUFFIX = "analyzed_plan: {input}\nexplain:"

_plan1 = """
GlobalLimit 100
    +- LocalLimit 100
       +- Sort [d_year#778 ASC NULLS FIRST, sum_agg#743 DESC NULLS LAST, brand_id#741 ASC NULLS FIRST], true
          +- Aggregate [d_year#778, i_brand#912, i_brand_id#911], [d_year#778, i_brand_id#911 AS brand_id#741, i_brand#912 AS brand#742, sum(ss_ext_sales_price#896) AS sum_agg#743]
             +- Filter (((d_date_sk#772 = ss_sold_date_sk#881) AND (ss_item_sk#883 = i_item_sk#904)) AND ((i_manufact_id#917 = 128) AND (d_moy#780 = 11)))
                +- Join Inner
                   :- Join Inner
                   :  :- SubqueryAlias dt
                   :  :  +- SubqueryAlias main.tpcds_sf1_delta.date_dim
                   :  :     +- Relation main.tpcds_sf1_delta.date_dim[d_date_sk#772,d_date_id#773,d_date#774,d_month_seq#775,d_week_seq#776,d_quarter_seq#777,d_year#778,d_dow#779,d_moy#780,d_dom#781,d_qoy#782,d_fy_year#783,d_fy_quarter_seq#784,d_fy_week_seq#785,d_day_name#786,d_quarter_name#787,d_holiday#788,d_weekend#789,d_following_holiday#790,d_first_dom#791,d_last_dom#792,d_same_day_ly#793,d_same_day_lq#794,d_current_day#795,... 4 more fields] parquet
                   :  +- SubqueryAlias main.tpcds_sf1_delta.store_sales
                   :     +- Relation main.tpcds_sf1_delta.store_sales[ss_sold_date_sk#881,ss_sold_time_sk#882,ss_item_sk#883,ss_customer_sk#884,ss_cdemo_sk#885,ss_hdemo_sk#886,ss_addr_sk#887,ss_store_sk#888,ss_promo_sk#889,ss_ticket_number#890L,ss_quantity#891,ss_wholesale_cost#892,ss_list_price#893,ss_sales_price#894,ss_ext_discount_amt#895,ss_ext_sales_price#896,ss_ext_wholesale_cost#897,ss_ext_list_price#898,ss_ext_tax#899,ss_coupon_amt#900,ss_net_paid#901,ss_net_paid_inc_tax#902,ss_net_profit#903] parquet
                   +- SubqueryAlias main.tpcds_sf1_delta.item
                      +- Relation main.tpcds_sf1_delta.item[i_item_sk#904,i_item_id#905,i_rec_start_date#906,i_rec_end_date#907,i_item_desc#908,i_current_price#909,i_wholesale_cost#910,i_brand_id#911,i_brand#912,i_class_id#913,i_class#914,i_category_id#915,i_category#916,i_manufact_id#917,i_manufact#918,i_size#919,i_formulation#920,i_color#921,i_units#922,i_container#923,i_manager_id#924,i_product_name#925] parquet
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
Given a pyspark dataframe `df`.
The output columns of `df`:
{columns}

{explain}

Now help me write python code to visualize the result of `df` using plotly:
1. All the code MUST be in one code block.
2. There is no need to install any package with pip.
3. Show the plot directly, instead of saving into a HTML
"""

PLOT_PROMPT = PromptTemplate(
    input_variables=["columns", "explain"], template=PLOT_PROMPT_TEMPLATE
)
