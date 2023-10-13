import contextlib
import hashlib
import io
import json
import logging
import pandas as pd
from io import StringIO
from pyspark.sql import SparkSession
from pyspark_ai import SparkAI
from pyspark_ai.ai_utils import AIUtils

# Constants
INCLUDE_KEYS = [
    "x", "y", "xaxis", "yaxis", "type", "orientation",
    "lat", "lon", "z",  # density_mapbox
    "domain", "labels", "values",  # pie # labels should not be compared
    "xbingroup", "ybingroup",  # bin
]

DATASETS = [
    "https://raw.githubusercontent.com/plotly/datasets/master/1962_2006_walmart_store_openings.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/2011_february_aa_flight_paths.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/US-shooting-incidents.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/titanic.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/winequality-red.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv",
]


def generate_id(dataset, description):
    combined_string = dataset + description
    return hashlib.md5(combined_string.encode()).hexdigest()


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


def filter_golden_json_data(json_string, include_keys=None):
    data_dict = json.loads(json_string)

    if 'data' not in data_dict:
        raise ValueError("'data' field does not exist in the provided JSON string.")
    data_content = data_dict['data'][0]

    if include_keys is None:
        filtered_data = {key: data_content[key] for key in data_content}
    else:
        filtered_data = {key: data_content[key] for key in include_keys if key in data_content}

    return {'data': filtered_data}


def gen_test_case(dataset):
    """
    Generate test cases with uuid -> test.json
    """
    if "1962_2006_walmart_store_openings.csv" in dataset:
        easy_descriptions = [
            "Line plot displaying the number of Walmart stores opened each year from 1962 to 2006.",
            "Bar plot showing the count of Walmart stores by state.",
            "Histogram illustrating the distribution of store openings by month.",
            "Boxplot representing the distribution of store conversions.",
            "Density plot showcasing the distribution of Walmart store openings over the years.",
            "Area plot highlighting the cumulative number of Walmart stores opened from 1962 to 2006.",
            "Scatter plot of Walmart store locations using latitude and longitude.",
            "Pie chart showing the proportion of different types of Walmart stores.",
            "Bar plot representing the number of Walmart stores opened each month.",
            "Line plot illustrating the trend of store conversions over the years.",
            "Hexagonal bin plot showing the density of Walmart store locations using latitude and longitude.",
            "Scatter plot with a color gradient representing the year of opening, plotted against latitude and longitude.",
            "Boxplot comparing the distribution of store openings by month, separated by store type.",
            "Density plot illustrating the distribution of store conversions, segmented by state.",
            "Area plot representing the cumulative number of Walmart stores, with each type of store shaded differently."
        ]

        hard_descriptions = [
            "Yearly trend of Walmart store openings from 1996 to 2006.",
            "Top 5 states with the highest number of Walmart store openings.",
            "Proportion of stores by type after 2000.",
            "Proportion of stores by type before 2000.",
            "Distribution of the number of stores opened each year after 2000.",
        ]
    elif "2011_february_aa_flight_paths.csv" in dataset:
        easy_descriptions = [
            'Bar plots showcasing the frequency of AA flights from different starting airports.',
            'Bar plots representing the distribution of ending locations for AA flights.',
            'Histogram illustrating the common starting longitudes for the flights.',
            'Histogram depicting the popular ending latitudes for AA flights.',
            'Boxplot summarizing the range of starting latitudes for all AA flights.',
            'Boxplot visualizing the range of ending longitudes for these flights.',
            'Density plots highlighting the concentration of starting locations.',
            'Density plots emphasizing the concentration of ending locations.',
            'Area plots comparing the frequency of flights based on starting and ending latitudes.',
            'Scatter plots visualizing the correlation between starting and ending locations for the flights.',
            'Pie plots representing the proportion of flights based on their starting airports.'
        ]

        hard_descriptions = [
            "Top 5 airports with the highest number of arrivals.",
            "Top 5 airports with the highest number of departure.",
            "Proportion of top 5 flights based on their starting airports.",
        ]
    elif "2011_us_ag_exports.csv" in dataset:
        easy_descriptions = [
            "Bar plot comparing beef exports across states.",
            "Histogram showing the distribution of poultry exports.",
            "Boxplot representing the spread of dairy exports.",
            "Area plot indicating the trend of wheat exports by state.",
            "Hexagonal bin plot showing the relationship between fruits fresh and fruits processed exports.",
            "Pie chart representing the share of total veggies exported by each state.",
            "Bar plot of pork exports for each state.",
            "Histogram of beef exports distribution.",
            "Boxplot of wheat exports for each state.",
            "Hexagonal bin plot for the relationship between beef and pork exports.",
            "Pie chart of the share of dairy exports by state.",
            "Bar plot comparing cotton exports across states.",
            "Hexagonal bin plot illustrating the correlation between fresh and processed fruit exports.",
            "Boxplot showcasing the range and outliers of total exports for the southern states.",
            "Area plot stacked by category showing the trend of exports for a selected group of states.",
            "Hexagonal bin plot contrasting the exports of fresh fruits against dairy products.",
        ]

        hard_descriptions = [
            "Dairy earnings for the top 5 states.",
            "Meat revenue breakdown for New York state.",
            "Proportion of the export market for fresh versus processed fruits.",
            "Relationship between beef revenue and poultry revenue among states: Texas, California, Florida, New York, and Illinois.",
            "Proportion of meat revenue components for Texas."
        ]
    elif "US-shooting-incidents.csv" in dataset:
        easy_descriptions = [
            "Bar plot showing the number of incidents per state.",
            "Line plot showing the trend of incidents over the years.",
            "Histogram of the number of incidents per year.",
            "Pie chart representing the distribution of causes of incidents.",
            "Scatter plot of latitude versus longitude to visualize the locations of incidents.",
            "Bar plot showing the number of incidents involving canines.",
            "Area plot indicating the number of incidents over the years.",
            "Boxplot of incidents per year to visualize the distribution.",
            "Histogram showing the distribution of incidents based on cause.",
            "Bar plot showing the number of incidents for each cause in short form.",
            "Scatter plot showing the distribution of incidents based on latitude and longitude.",
            "Area plot representing the number of incidents involving canines over the years.",
            "Hexagonal bin plot of latitude versus longitude to visualize incident density.",
            "Pie chart representing the distribution of incidents based on whether it involved a canine or not.",
            "Histogram showing the distribution of incidents based on state.",
            "Line plot indicating the trend of incidents based on cause over the years.",
            "Scatter plot showing the distribution of incidents in different states based on latitude and longitude.",
            "Boxplot showing the distribution of incidents based on cause.",
            "Area plot representing the number of incidents in each state over the years.",
        ]
        hard_descriptions = [
            "Yearly trend of incidents.",
            "Top 5 departments with the highest incidents in 2001.",
            "Proportion of incidents by top 5 casuse of death.",
            "Top 5 casuse of death in 2001.",
            "Cumulative incidents from 2000 to 2005.",
        ]
    elif "titanic.csv" in dataset:
        easy_descriptions = [
            "Bar plot showing the number of passengers in each class.",
            "Histogram showcasing the age distribution of passengers.",
            "Pie chart representing the gender distribution aboard the Titanic.",
            "Bar plot indicating the number of survivors and non-survivors.",
            "Area plot illustrating the fare distribution over different passenger classes.",
            "Scatter plot of age against fare to see if there's a correlation.",
            "Bar plot showing the number of passengers boarding from each embarkation port.",
            "Histogram displaying the distribution of fares paid by passengers.",
            "Pie chart showcasing the distribution of passengers in lifeboats.",
            "Bar plot indicating the number of siblings/spouses each passenger had aboard.",
            "Boxplot showing the fare distribution for male and female passengers.",
            "Area plot representing the age distribution of survivors and non-survivors.",
            "Scatter plot of siblings/spouses against parents/children to see family size.",
            "Bar plot showcasing the number of passengers with and without cabins.",
            "Pie chart representing the distribution of passengers based on their embarkation port.",
            "Bar plot showing the survival rate for each gender.",
            "Boxplot displaying the age distribution for survivors and non-survivors.",
            "Area plot illustrating the fare distribution over different embarkation ports.",
            "Scatter plot of age against the number of siblings/spouses.",
            "Bar plot indicating the number of parents/children each passenger had aboard.",
            "Histogram showcasing the age distribution of male and female passengers.",
            "Pie chart representing the survival rate for each passenger class.",
            "Bar plot showing the number of passengers in each lifeboat."
        ]

        hard_descriptions = [
            "Distribution of age for survivors and non-survivors.",
            "Trend of average fare over age groups.",
            "Number of survivors from each embarkation port.",
            "Fare variability across ticket classes.",
            "Cumulative number of passengers across age groups.",
            "Proportion of passengers by ticket class."
        ]
    elif "winequality-red.csv" in dataset:
        easy_descriptions = [
            "Histogram of alcohol percentages in the wine samples.",
            "Bar plot of average salt content in different quality wines.",
            "Area plot representing the amount of residual sugar across different wine samples.",
            "Pie chart showing the proportion of wines with different pH levels.",
            "Histogram showcasing the distribution of sulfur dioxide levels.",
            "Bar plot indicating the average citric acid content in wines of varying quality.",
            "Area plot representing the distribution of sulphate levels across wine samples.",
            "Pie chart illustrating the proportion of wines in each quality category.",
            "Histogram of the distribution of pH values in the wine samples.",
            "Bar plot showing average alcohol content for each wine quality score.",
            "Scatter plot comparing salt content with wine quality scores.",
            "Pie chart representing the proportion of wines with different levels of fixed acidity.",
            "Bar plot of average volatile acidity levels for different wine quality scores.",
            "Boxplot showing the range of residual sugar levels in the wine samples.",
            "Pie chart showing the proportion of wines with varying levels of citric acid.",
            "Histogram illustrating the distribution of sulphate level [0.4, 0.6]",
            "Scatter plot comparing fixed acidity 12-14 with volatile acidity [0.2, 0.4]",
            "Area plot representing the distribution of total sulfur dioxide levels [30, 50]",
            "Histogram of the distribution of density values above 1 in the wine samples",
            "Scatter plot comparing residual sugar content from 6 to 8 with alcohol percentages from 9 to 10",
        ]

        hard_descriptions = [
            "Trend of average fixed acidity levels for wines rated from 3 to 6",
            "Trend of average volatile acidity levels for wines rated from 3 to 6",
            "Distribution of density for wines rated 8.",
            "Distribution of pH levels for wines rated 8.",
            "Distribution of citric acid levels for wines rated 8.",
        ]
    elif "us-cities-top-1k.csv" in dataset:
        easy_descriptions = [
            "Bar plot of the top 5 most populous cities",
            "Area plot of populations for the top 5 cities",
            "Scatter plot of latitude versus longitude for top 5 most populous cities",
            "Pie chart showing the population distribution of the top 5 cities",
            "Bar plot of the number of cities in each state",
            "Histogram of latitudes of the top 5 most populous cities",
            "Scatter plot of population versus latitude for the top 10 most populous cities",
            "Hexagonal bin plot of latitude versus longitude for the top 5 most populous cities",
            "Bar plot showing populations of the top 5 most populous cities in California",
            "Boxplot of populations for the top 5 most populous cities in Texas",
            "Area plot of populations for the top 5 most populous cities in Florida",
            "Pie chart of the number of cities in the top 5 states",
            "Bar plot of the least populous 8 cities",
            "Boxplot of latitudes for cities in New York state",
            "Area plot of populations for the bottom 10 cities",
            "Hexagonal bin plot of population versus latitude for the top 5 most populous cities",
            "Mapbox density map showing the top 10 most populous cities"
            "Mapbox density map showing the top 5 most populous cities in California",
            "Pie chart showing the population distribution of most populous city in each of the top 5 states",
        ]

        hard_descriptions = [
            "Top 10 cities by population",
            "Population distribution of cities in Oklahoma",
            "Relationship between latitude and population for top 10 cities by population",
            "Cities with a population greater than 1 million",
            "Population comparison of cities in South Carolina and Nevada"
        ]
    else:
        raise ValueError("No automation of test cases curation for the given dataset.")

    combined_list = []
    for desc in easy_descriptions:
        item = {
            "uuid": generate_id(dataset, desc),
            "dataset": dataset,
            "language": "English",
            "complexity": "easy",
            "description": desc
        }
        combined_list.append(item)

    for desc in hard_descriptions:
        item = {
            "uuid": generate_id(dataset, desc),
            "dataset": dataset,
            "language": "English",
            "complexity": "hard",
            "description": desc
        }
        combined_list.append(item)

    # Convert the combined list to JSON
    with open('benchmark/plot/data/train/dummy-train.json', 'w') as file:
        json.dump(combined_list, file, indent=4)

def substitute_show_to_json(string):
    import re

    return re.sub(r'(\w+)\.show\(\)', r'print(\1.to_json())', string)


def gen_golden(dataset):
    """
    Generate:
        - code that generates golden plots -> train-code.json
        - golden plots in json -> train-golden.json
    """
    spark = SparkSession.builder.getOrCreate()
    spark_ai = SparkAI(spark_session=spark, verbose=True)
    spark_ai.activate()

    pdf = pd.read_csv(dataset)
    df = spark_ai._spark.createDataFrame(pdf)

    include_keys = INCLUDE_KEYS
    golden_codes = []
    golden_jsons = []

    with open('benchmark/plot/data/train/dummy-train.json', 'r') as file:
        test_cases = json.load(file)

    for test_case in test_cases:
        if test_case['dataset'] == dataset:
            uuid = test_case['uuid']
            try:
                original_code = df.ai.plot(test_case['description'])
                code = substitute_show_to_json(original_code)

                # captured_output = capture_plot_logs(df, test_case['description'], spark_ai)
                # codeblocks = AIUtils.extract_code_blocks(captured_output)
                # sub_codeblocks = substitute_show_to_json(codeblocks)
                # code = "\n".join(sub_codeblocks)

                golden_codes.append({
                    'uuid': uuid,
                    'code': code
                })

                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    exec(compile(code, "plot_df-CodeGen-benchmark", "exec"))
                captured_output = buffer.getvalue()[:-1]

                # Take data field (not layout field) as golden
                golden_jsons.append({
                    'uuid': uuid,
                    'plot_json': filter_golden_json_data(captured_output, include_keys)
                })
            except Exception as e:
                print(f"Test case with UUID {uuid} failed due to: {str(e)}")
                continue

    # Convert the golden_codes list to JSON
    with open('benchmark/plot/data/train/dummy-train-code.json', 'w') as file:
        json.dump(golden_codes, file, indent=4)

    # Convert the golden_jsons list to JSON
    with open('benchmark/plot/data/train/dummy-train-golden.json', 'w') as file:
        json.dump(golden_jsons, file, indent=4)


if __name__ == "__main__":
    dataset = "https://raw.githubusercontent.com/plotly/datasets/master/1962_2006_walmart_store_openings.csv"
    gen_test_case(dataset)
    gen_golden(dataset)
