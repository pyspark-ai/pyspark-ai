import json


def extract_subset(fig_dict, keys):
    data = fig_dict.get("data")

    # Check if "data" is a list
    if isinstance(data, list):
        subset_data = [{key: entry.get(key) for key in keys} for entry in data]
    # Check if "data" is a dictionary
    elif isinstance(data, dict):
        subset_data = {key: data.get(key) for key in keys}
    else:
        raise ValueError("The 'data' field is neither a list nor a dictionary.")

    return {"data": subset_data}

def f1(df):
    # Import necessary libraries
    from pyspark.sql import SparkSession
    import plotly.graph_objects as go
    import pandas as pd

    # Start Spark session
    spark = SparkSession.builder.appName('SparkSQL').getOrCreate()

    # Assuming df is already defined and contains the data
    # df = spark.sql("SELECT * FROM LogicalRDD")

    # Convert Spark DataFrame to Pandas DataFrame
    pandas_df = df.toPandas()

    # Create bar plot of daily closing prices
    fig = go.Figure(data=[go.Bar(x=pandas_df['DATE'], y=pandas_df['CLOSE'])])

    # Set plot title and labels
    fig.update_layout(title_text='Daily Closing Prices', xaxis_title='Date',
                      yaxis_title='Closing Price')

    # Display the plot
    fig_dict = json.loads(fig.to_json())
    return extract_subset(fig_dict, ['x', 'y', 'type'])


def f2(df):
    # Import necessary libraries
    from pyspark.sql import SparkSession
    import plotly.graph_objects as go
    import pandas as pd

    # Start Spark session
    spark = SparkSession.builder.appName('SparkSQL').getOrCreate()

    # Assuming df is already a Spark DataFrame
    # Calculate daily price changes
    df = df.withColumn('DAILY_CHANGE', df['CLOSE'] - df['OPEN'])

    # Convert Spark DataFrame to Pandas DataFrame
    pandas_df = df.select("*").toPandas()

    # Create histogram
    fig = go.Figure(data=[go.Histogram(x=pandas_df['DAILY_CHANGE'], nbinsx=100)])

    # Set layout
    fig.update_layout(
        title_text='Histogram of Daily Price Changes',
        xaxis_title_text='Price Change',
        yaxis_title_text='Count',
        bargap=0.2,
        bargroupgap=0.1
    )

    # Display the plot
    fig_dict = json.loads(fig.to_json())
    return extract_subset(fig_dict, ['x', 'type'])


def f3(df):
    # Import necessary libraries
    from pyspark.sql import SparkSession
    import pandas as pd
    import plotly.express as px

    # Start Spark session
    spark = SparkSession.builder.appName('SparkSQL').getOrCreate()

    # Assuming df is already a Spark DataFrame
    # If not, convert it to Spark DataFrame
    # df = spark.createDataFrame(df)

    # Aggregate df
    df_agg = df.groupBy('DATE').agg({'CLOSE': 'mean'})

    # Convert Spark DataFrame to Pandas DataFrame
    df_pd = df_agg.toPandas()

    # Create box plot of daily closing prices
    fig = px.box(df_pd, y='avg(CLOSE)', labels={'avg(CLOSE)': 'Daily Closing Prices'})

    # Display the plot
    fig_dict = json.loads(fig.to_json())
    return extract_subset(fig_dict, ['y', 'type'])


def f4(df):
    # Import necessary libraries
    from pyspark.sql import SparkSession
    import pandas as pd
    import plotly.graph_objects as go

    # Start Spark session
    spark = SparkSession.builder.appName('SparkSQL').getOrCreate()

    # Assuming df is already a Spark DataFrame
    # Perform any necessary aggregation here

    # Convert Spark DataFrame to Pandas DataFrame
    pandas_df = df.toPandas()

    # Create line plot of daily opening prices over time
    fig = go.Figure(data=go.Scatter(x=pandas_df['DATE'], y=pandas_df['OPEN'], mode='lines'))

    # Set plot title and labels
    fig.update_layout(title='Daily Opening Prices Over Time',
                      xaxis_title='Date',
                      yaxis_title='Opening Price')

    # Display the plot
    fig_dict = json.loads(fig.to_json())
    return extract_subset(fig_dict, ['mode', 'x', 'y', 'type'])


def f5(df):
    # Import necessary libraries
    from pyspark.sql import SparkSession
    import plotly.express as px
    import pandas as pd

    # Start Spark session
    spark = SparkSession.builder.appName('SparkSQL').getOrCreate()

    # Assuming df is already defined and loaded with data
    # Aggregate df if necessary (not required in this case)

    # Convert Spark DataFrame to Pandas DataFrame
    pandas_df = df.toPandas()

    # Create scatter plot using plotly
    fig = px.scatter(pandas_df, x='OPEN', y='CLOSE')

    # Display the plot
    fig_dict = json.loads(fig.to_json())
    return extract_subset(fig_dict, ['mode', 'x', 'y', 'type'])


def f6(df):
    # Import necessary libraries
    from pyspark.sql import SparkSession
    import pandas as pd
    import plotly.express as px

    # Start Spark session
    spark = SparkSession.builder.appName('SparkSQL').getOrCreate()

    # Filter the DataFrame to get only the days with opening prices above 190.00
    df_filtered = df.filter(df.OPEN > 190.00)

    # Aggregate the DataFrame to get the count of days with opening prices above 190.00
    df_agg = df_filtered.groupBy('DATE').count()

    # Convert the Spark DataFrame to a pandas DataFrame
    df_pandas = df_agg.toPandas()

    # Create a pie chart using plotly
    fig = px.pie(df_pandas, values='count', names='DATE',
                 title='Days with opening prices above 190.00')

    # Display the plot
    fig_dict = json.loads(fig.to_json())
    return extract_subset(fig_dict, ['labels', 'values', 'type'])


def f7(df):
    # Import necessary libraries
    from pyspark.sql import SparkSession
    import plotly.graph_objects as go
    import pandas as pd

    # Start Spark session
    spark = SparkSession.builder.appName('AreaChart').getOrCreate()

    # Assuming df is already defined and loaded with data
    # df = spark.read...

    # Convert Spark DataFrame to Pandas DataFrame
    pandas_df = df.toPandas()

    # Sort the data by date
    pandas_df = pandas_df.sort_values('DATE')

    # Create area chart
    fig = go.Figure(data=go.Scatter(x=pandas_df['DATE'], y=pandas_df['CLOSE'], fill='tozeroy'))

    # Set plot title and labels
    fig.update_layout(title='Area Chart of Daily Closing Prices', xaxis_title='Date',
                      yaxis_title='Closing Price')

    # Display the plot
    fig_dict = json.loads(fig.to_json())
    return extract_subset(fig_dict, ['fill', 'x', 'y', 'type'])


def f8(df):
    # Import necessary libraries
    from pyspark.sql import SparkSession
    import plotly.graph_objects as go
    import pandas as pd

    # Start Spark session
    spark = SparkSession.builder.appName('SparkSQL').getOrCreate()

    # Assuming df is already defined and contains the data
    # Calculate daily high-low range
    df = df.withColumn('RANGE', df['HIGH'] - df['LOW'])

    # Convert Spark DataFrame to Pandas DataFrame
    pandas_df = df.select('DATE', 'RANGE').toPandas()

    # Create bar plot using plotly
    fig = go.Figure(data=go.Bar(x=pandas_df['DATE'], y=pandas_df['RANGE']))
    fig.update_layout(title_text='Daily High-Low Range', xaxis_title='Date', yaxis_title='Range')
    fig_dict = json.loads(fig.to_json())
    return extract_subset(fig_dict, ['x', 'y', 'type'])


def f9(df):
    # Import necessary libraries
    from pyspark.sql import SparkSession
    import pandas as pd
    import plotly.express as px

    # Start Spark session
    spark = SparkSession.builder.appName('SparkSQL').getOrCreate()

    # Assuming df is already a Spark DataFrame
    # Aggregate the data
    df_agg = df.groupBy("DATE").sum("VOLUME")

    # Convert the Spark DataFrame to a pandas DataFrame
    df_pd = df_agg.toPandas()

    # Create a histogram of daily trading volume
    fig = px.histogram(df_pd, x="sum(VOLUME)", nbins=50,
                       labels={'sum(VOLUME)': 'Daily Trading Volume'})

    # Display the plot
    fig_dict = json.loads(fig.to_json())
    return extract_subset(fig_dict, ['x', 'type'])


def f10(df):
    # Import necessary libraries
    from pyspark.sql import SparkSession
    import plotly.graph_objects as go
    import pandas as pd

    # Start Spark session
    spark = SparkSession.builder.appName('pyspark_plot').getOrCreate()

    # Assuming df is already a Spark DataFrame
    # If not, you can create it using spark.createDataFrame()

    # Aggregate the data
    df_agg = df.groupBy("DATE").sum("VOLUME")

    # Convert the Spark DataFrame to a pandas DataFrame
    df_pd = df_agg.toPandas()

    # Sort the DataFrame by date
    df_pd = df_pd.sort_values('DATE')

    # Create a line plot of daily trading volume change
    fig = go.Figure(data=go.Scatter(x=df_pd['DATE'], y=df_pd['sum(VOLUME)'], mode='lines'))

    # Set plot title and labels
    fig.update_layout(title='Daily Trading Volume Change',
                      xaxis_title='Date',
                      yaxis_title='Volume')

    # Display the plot
    fig_dict = json.loads(fig.to_json())
    return extract_subset(fig_dict, ['mode', 'x', 'y', 'type'])
