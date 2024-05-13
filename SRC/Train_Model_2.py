import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Read Cleaned CSV Files


# Define the independent variables (features) and the target variable


# Convert categorical variable in the X dataset into dummy variables


# Split the data into training and testing sets


# ================================================== B. TRAIN MODEL 2 ================================================== #

# New Model Name
# Build the model


# Train the model


# Evaluate the model on the testing data


"""
#Answers: 


#Insight Gained:
    - 
"""

# Save the predictions to a CSV file for Model 2


# Create dummy feature importance values


# Save feature importance values to CSV files



# -------------------------------------------------------
# New Model Name
# Build the model

"""
Answers:

"""

# Hyperparameters the  model ....


# Rebuild the Model


# Calculate the mean validation accuracy score


"""
Answer:


    Mean validation accuracy score: 
"""


# Preprocess the test data in the same way as the training data


# Make sure the processed test data has the same columns as the training data


# Now you can make predictions on the processed test data


# Calculate the mean validation accuracy score


"""
#Answer: Mean validation accuracy score: 

Insight Gained:
    - 
"""

# -------------------------------------------------------
# New Model Name
# Build the model

# Calculate the mean validation accuracy score


"""
Answers:


    Mean validation accuracy score: 
"""


# Hyperparameter Tuning for .....


# Rebuild the Model


# Calculate the mean validation accuracy score


"""
Answer:


    Mean validation accuracy score: 
"""


# Preprocess the test data in the same way as the training data


# Make sure the processed test data has the same columns as the training data


# Now you can make predictions on the processed test data


""" 
Insight Gained:
    - 
"""

# -------------------------------------------------------
# Take the model with the best accuracy score
# Fill 'Loan_Status' with predictions


# Replace 0 and 1 with 'N' and 'Y'


# Convert the submission DataFrame to a .csv format


# Convert the importances into a pandas DataFrame


# Plot the feature importances


"""
# Insight Gained:
    - 
"""

# Save Model 2 as a pickle file
