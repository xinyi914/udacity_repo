# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project aimed to practice cleaning code and testing for data science project. We tried to identify the curtomer churn based on given data frame. We did eda and then used logistic regression and random forest models to do the prediction. We finally plot the classification report, roc curves, feature importance plot of the result.

## Files and data description
churn_library.py contains main functions to find customers who are likely to churn.

churn_script_logging_and_tests.py contains unit tests for each function in the churn_library.py.

Churn_notebook.ipynb is the original file containing whole ML process.

Plots for eda are stored in images/eda
Plots for results are stored in images/results

Our data is in the data/bank_data.csv.

Here is the structure of the project:
<pre> .
├── Guide.ipynb          # Given: Getting started and troubleshooting tips
├── churn_notebook.ipynb # Given: Contains the code to be refactored
├── churn_library.py     # ToDo: Define the functions
├── churn_script_logging_and_tests.py # ToDo: Finish tests and logs
├── README.md            # ToDo: Provides project overview, and instructions to use the code
├── data                 # Read this data
│   └── bank_data.csv
├── images               # Store EDA results 
│   ├── eda
│   └── results
├── logs                 # Store logs
└── models               # Store models
</pre>

## Running Files
To run files:

First, create a virtual environment:
<pre>conda create --name churn_predict python=3.8</pre>
<pre>conda activate churn_predict</pre>

Second, install the packages:
<pre>python -m pip install -r requirements_py3.8.txt</pre>

Third, run unit tests:
<pre>python churn_script_logging_and_tests.py</pre>

Finally, run model:
<pre>python churn_library.py</pre>

Check the plot of result under images/results



