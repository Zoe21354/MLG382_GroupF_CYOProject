import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Read Cleaned CSV Files
cleaned_raw_data = pd.read_csv('Data/Feature Engineering/new_features_engineered.csv')
cleaned_raw_data_copy = cleaned_raw_data.copy()

# Define the independent variables (features) and the target variable
X = cleaned_raw_data_copy.drop('loan_status', axis=1)
y = cleaned_raw_data_copy['loan_status']

# Convert categorical variable in the X dataset(all columns except 'loan_status') into dummy variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create new DataFrames for training and testing sets
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Save the training and testing sets to CSV files
train_data.to_csv('Data/Split Data/Model 2/train_data_ML2.csv', index=False)
test_data.to_csv('Data/Split Data/Model 2/test_data.ML2.csv', index=False)


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
