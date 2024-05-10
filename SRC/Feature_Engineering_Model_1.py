import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt                         # Matplotlib and Seaborn is used to create visual graphs
import seaborn as sns  
import warnings                                         # Ignores any future warnings
warnings.filterwarnings('ignore')  


train_data= pd.read_csv('Data/Split Data/train_data.csv')
test_data=pd.read_csv('Data/Split Data/test_data.csv')

# Convert the 'Person_Emp_Length' column to 'object'


# =============================================== FEATURE ENGINEERING MODEL 1 =============================================== #
"""     
    # Feature engineering transforms or combines raw data into a format that can be easily understood by machine learning models.
    # Creates predictive model features, also known as a dimensions or variables, to generate model predictions.
    # This highlights the most important patterns and relationships in the data, which then assists the machine learning model to learn from the data more effectively.
"""

# Feature 1: 


# Feature 2: 

# Feature 3: 


# Remove all features that created the new features
    # The correlation between those old feature and the new features are very high.
    # Due to this the excess noise in the datasets are removed.


# Check the coloumns were dropped


"""
Training Data Columns: 
    

Testing Data  Columns:

"""

# Store new Features in CSV files
