
# Import Libraries
import csv                                              # Read and Write to CSV files
import pandas as pd                                     # Manipulation and analysis of data
import numpy as np                                      # Mathematical operations
import matplotlib.pyplot as plt                         # Matplotlib and Seaborn is used to create visual graphs
import seaborn as sns                                   
from sklearn.model_selection import train_test_split    # Splits the raw_data into two sets of data
import warnings                                         # Ignores any future warnings
warnings.filterwarnings('ignore')   


# Read Unclean CSV Files
raw_data = pd.read_csv('Data/Original Data/credit_risk_raw_data.csv')
raw_data_copy = raw_data.copy()

validation_data = pd.read_csv('Data/Original Data/validation_data.csv')
validation_data_copy = validation_data.copy()


# ======================================================= A. DATA ANALYSIS PROCESSES ======================================================= #

# 1. Dataset Analysis
# Dataset Attributes:
print(f"Raw Data Columns: {raw_data_copy.columns}\n")
print(f"Validation Data Columns:{validation_data_copy.columns}\n")

"""
# Answer:
    - Raw Data Columns: 
        Index(['person_age', 'person_income', 'person_home_ownership', 'person_emp_length', 'loan_intent', 'loan_amnt', 'loan_int_rate',        
            'loan_percent_income', 'cb_person_cred_hist_length', 'cb_person_default_on_file', 'loan_status'],dtype='object')

    - Validation Data Columns:
        Index(['person_age', 'person_income', 'person_home_ownership','person_emp_length', 'loan_intent', 'loan_amnt', 'loan_int_rate',
            'loan_percent_income', 'cb_person_cred_hist_length','cb_person_default_on_file'],dtype='object')

# Insights Gained:
    - The attribute names are inconsistent and will need standardizing in the data processing section.
    - Feature Variable (Independent variable): This variable stands alone and is not changed by other variables that are being measured. It is denoted as X in ML algorithms.
    - Target Variable (Dependent variable): This is the variable that is to be predicted. It is often denoted as Y in ML algorithms.
    - In both datasets there are 10 feature variables but only the raw_data dataset has 1 target variable.
    - The target variable in the raw_data dataset is the loan_status attribute.
    - This variable will be predicted using models for the validation_data dataset.
"""

# Dataset DataTypes:
print(f"Raw Dataset Datatypes:\n{raw_data_copy.dtypes}\n")
print(f"Validation Dataset Datatypes:\n{validation_data_copy.dtypes}\n")

"""
#Answers:
    - Raw Dataset Datatypes:
        person_age                      int64
        person_income                   int64
        person_home_ownership          object
        person_emp_length             float64
        loan_intent                    object
        loan_amnt                       int64
        loan_int_rate                 float64
        loan_percent_income           float64
        cb_person_cred_hist_length      int64
        cb_person_default_on_file      object
        loan_status                     int64
        dtype: object

    - Validation Dataset Datatypes:
        person_age                      int64
        person_income                   int64
        person_home_ownership          object
        person_emp_length               int64
        loan_intent                    object
        loan_amnt                       int64
        loan_int_rate                 float64
        loan_percent_income           float64
        cb_person_cred_hist_length      int64
        cb_person_default_on_file      object
        dtype: object

# Insights Gained:
    - There is a discrepancy between the two datasets: the "person_emp_length" attribute is of datatype float64 in the raw_data.csv file but of datatype object in the validation_data.csv file. 
    - This could lead to potentially issues when modeling, as the model might be expecting the same data type for a given attribute.
    - This discrepancy will need to be fixed in the data processing section.
"""

# Dataset Shape:
print(f"Raw Data Shape:\n{raw_data_copy.shape}")
print(f"Validation Data Shape:\n{validation_data_copy.shape}")

"""
# Answers:
    - Raw Data Shape: (1457, 11)
    - Validation Data Shape: (470, 10)

# Insights Gained:
    - Raw Data Shape: 1457 rows and 11 columns
    - Validation Data Shape: 470 rows and 10 columns
"""

# 2. Univariate Analysis
"""
    # When there is just one variable in the data it is called univariate analysis. 
    # This is the most basic type of data analysis and finds patterns in the data.
    # Analyzing univariate data involves examining:
        - Frequency of data
        - Mean, mode, median, and range


    Nominal data: person_home_ownership, loan_intent, cb_person_default_on_file
"""
# Frequency and Bar charts of each Independent variable and the Dependant variable:
    #Get the count for each category in the variable
    #Normalize the data to get the proportion of the different categories in the variable (each count is divided by the total number of values)
    #Plot a bar chart to visually display the data


#Dependent Variable
count = raw_data_copy['loan_status'].value_counts(normalize = True)
chart = count.plot.bar(title = 'Loan Status', xlabel = 'Categories', ylabel = 'Frequency')
for i, v in enumerate(count):
    chart.text(i, v + 0.01, str(round(v, 2)), ha='center', va='bottom')
plt.show()

"""
#Insight Gained:
    - 
"""


#Independent Variable (Ordinal)
#loan_percent_income
plt.figure(1)
plt.subplot(121)
raw_data_copy.dropna()
sns.distplot(raw_data_copy['loan_percent_income'])
plt.title('Distribution of Loan Income Percentage')
plt.xlabel('Loan Income Percentage')
plt.ylabel('Density')
plt.subplot(122)
boxplot =raw_data_copy['loan_percent_income'].plot.box()
boxplot.set_title('Box Plot of Loan Income Percentage')
boxplot.set_xlabel('Density')
plt.show()

"""
#Insight Gained:
    - 
"""

#loan_int_rate
plt.figure(1)
plt.subplot(121)
raw_data_copy.dropna()
sns.distplot(raw_data_copy['loan_int_rate'])
plt.title('Distribution of Loan Interest Rate')
plt.xlabel('Loan Rate')
plt.ylabel('Density')
plt.subplot(122)
boxplot =raw_data_copy['loan_int_rate'].plot.box()
boxplot.set_title('Box Plot of Loan Interest Rate')
boxplot.set_xlabel('Density')
plt.show()

"""
#Insight Gained:
    - 
"""


#person_income
plt.figure(1)
plt.subplot(121)
raw_data_copy.dropna()
sns.distplot(raw_data_copy['person_income'])
plt.title('Distribution of Income')
plt.xlabel('Loan Rate')
plt.ylabel('Density')
plt.subplot(122)
boxplot =raw_data_copy['person_income'].plot.box()
boxplot.set_title('Box Plot of Income')
boxplot.set_xlabel('Density')
plt.show()

"""
#Insight Gained:
    - 
"""

#loan_amnt
plt.figure(1)
plt.subplot(121)
raw_data_copy.dropna()
sns.distplot(raw_data_copy['loan_amnt'])
plt.title('Distribution of Loan Amount')
plt.xlabel('Loan Rate')
plt.ylabel('Density')
plt.subplot(122)
boxplot =raw_data_copy['loan_amnt'].plot.box()
boxplot.set_title('Box Plot of Loan Amount')
boxplot.set_xlabel('Density')
plt.show()

"""
#Insight Gained:
    - 
"""

#Independent Variable (Nominal)
#person_age
count = raw_data_copy['person_age'].value_counts('normalize = True')
chart = count.plot.bar(title='Age', xlabel = 'Categories', ylabel = 'Frequency')
for i, v in enumerate(count):
    chart.text(i, v + 0.01, str(round(v, 3)), ha='center', va='bottom')
plt.show()

"""
#Insight Gained:
    - The age group ranges from early 20s to 50s
    - With the largest number of people being of age 22 to 24
    - There are outlier ages that need to be addressed in the processing stage.
"""

#person_emp_length
count =raw_data_copy['person_emp_length'].value_counts('normalize = True')
chart = count.plot.bar(title='Employment Length', xlabel = 'Categories', ylabel = 'Frequency')
for i, v in enumerate(count):
    chart.text(i, v + 0.01, str(round(v, 3)), ha='center', va='bottom')
plt.show()

"""
#Insight Gained:
    - 
"""

#cb_person_cred_hist_length
count =raw_data_copy['cb_person_cred_hist_length'].value_counts('normalize = True')
chart = count.plot.bar(title='Credit History Length', xlabel = 'Categories', ylabel = 'Frequency')
for i, v in enumerate(count):
    chart.text(i, v + 0.01, str(round(v, 3)), ha='center', va='bottom')
plt.show()

"""
#Insight Gained:
    - 22.8% of the people who applied have a credit history length of 2
    - 20.5% of the people who applied have a credit history length of 4
    - 19.8% of the people who applied have a credit history length of 3
    - 6.1% of the people who applied have a credit history length of 14
    - 5.6% of the people who applied have a credit history length of 11
    - 5.2% of the people who applied have a credit history length of 12
    - 5.2% of the people who applied have a credit history length of 16
    - 5.1% of the people who applied have a credit history length of 15
    - 5.1% of the people who applied have a credit history length of 13
    - 4.6% of the people who applied have a credit history length of 17
    - The distribution appears to be right-skewed, meaning there are more individuals with shorter credit histories than those with longer ones.
"""

#person_home_ownership
count =raw_data_copy['person_home_ownership'].value_counts('normalize = True')
chart = count.plot.bar(title='Home Ownership', xlabel = 'Categories', ylabel = 'Frequency')
for i, v in enumerate(count):
    chart.text(i, v + 0.01, str(round(v, 3)), ha='center', va='bottom')
plt.show()

"""
#Insight Gained:
    - 58.5% of the people who apply for a loan pay rent
    - 34.2% of the people who apply for a loan pay a mortgage
    - 6.9% of the people who apply for a loan own a house
    - 0.3% of the people who apply for a loan have other living arrangements
"""

#loan_intent
count =raw_data_copy['loan_intent'].value_counts('normalize = True')
chart = count.plot.bar(title='Loan Intent', xlabel = 'Categories', ylabel = 'Frequency')
for i, v in enumerate(count):
    chart.text(i, v + 0.01, str(round(v, 3)), ha='center', va='bottom')
plt.show()

"""
#Insight Gained:
    - 19.6% of the people apply for a loan for their Education
    - 19.3% of the people apply for a loan for their Medical bills
    - 17.2% of the people apply for a loan for their Debt Consolidations
    - 16.5% of the people apply for a loan for Personal reason
    - 15.5% of the people apply for a loan for Venture funding
    - 11.9% of the people apply for a loan for Home Improvements
"""
