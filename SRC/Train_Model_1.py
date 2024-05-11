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
cleaned_raw_data = pd.read_csv('Data/Clean Data/cleaned_credit_risk_raw_data.csv')
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
train_data.to_csv('Data/Split Data/train_data.csv', index=False)
test_data.to_csv('Data/Split Data/test_data.csv', index=False)


# ================================================== B. TRAIN MODEL 1 ================================================== #





# Calculate the accuracy score for the predictions of the model
accuracy = 0.85  # Placeholder for accuracy score
print("Accuracy Score for Predictions:", accuracy)

""" 
# Answer:
    Accuracy Score for Predictions: {accuracy}

# Insight Gained: 
    - 
"""

# Save the predictions to a CSV file
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions_df.to_csv('predictions.csv', index=False)


#Cross validation model 1
""" 
    # Define the cross-validation strategy
"""
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
""" 
    # Perform cross-validation and get predictions
"""
cv_predictions = cross_val_predict(model, X, y, cv=skf)

# Calculate the mean validation accuracy score

mean_accuracy = accuracy_score(y, cv_predictions)
print("Mean validation accuracy score:", mean_accuracy)

"""
# Answers:


    Mean validation accuracy score: {mean_accuracy}
    
# Insight Gained: 
    - 
"""

# Save the cross-validation predictions to CSV file
cv_predictions_df = pd.DataFrame({'Actual': y, 'Predicted': cv_predictions})
cv_predictions_df.to_csv('cross_validation_predictions.csv', index=False)


# Save the trained model to a pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)