import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings                                         # Ignores any future warnings
warnings.filterwarnings('ignore')  


# ================================================== A. SPLITTING THE RAW DATA INFORMATION ================================================== #
"""     
    # Dummy data is used to convert the categorical data into 0's and 1's  to make it easy to be quantified and compared in the future models
        -> Example: Gender has Male and Female categories
        -> Using 'dummies' from pandas, it converts them into Gender_Male = 1 and Gender_Female = 0
    # Training data set has weight 80% 0r 0.8
    # Testing data set has weight 20% or 0.2
    # 'random_state=42' is used for reproducibility, meaning if the code is run multiple times the same train/test split will occur every time.
"""

# Read Cleaned CSV Files
cleaned_raw_data = pd.read_csv('Data/Cleaned Data/cleaned_raw_data.csv')
cleaned_raw_data_copy = cleaned_raw_data.copy()

# Convert the 'Person_Emp_Length' column to 'object'


# Define the independent variables (features) and the target variable



# Convert categorical variable in the X dataset(all columns except 'Loan_Status') into dummy variables


# Split the data into training and testing sets


# Create new DataFrames for training and testing sets


# Save the training and testing sets to CSV files



# ================================================== B. TRAIN MODEL 1 ================================================== #





# Calculate the accuracy score for the predictions of the model


""" 
# Answer:
    Accuracy Score for Predictions: 

# Insight Gained:
    - 
"""

# Save the predictions to a CSV file



#Cross validation model 1
""" 
    # 
"""

# Calculate the mean validation accuracy score

"""
# Answers:


    Mean validation accuracy score: 
    
# Insight Gained:
    - 
"""

# Save the cross-validation predictions to CSV file



# Save the trained model to a pickle file

