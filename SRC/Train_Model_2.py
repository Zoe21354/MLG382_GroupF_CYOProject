import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
submission = pd.DataFrame()
# Read Cleaned CSV Files
cleanedRaw = pd.read_csv('Data/Clean Data/cleaned_credit_risk_raw_data.csv')
cleanedRawCopy = cleanedRaw.copy()

# Define the independent variables (features) and the target variable
X = cleanedRawCopy.drop('loan_status', axis=1)
y = cleanedRawCopy['loan_status']

# Convert categorical variable in the X dataset into dummy variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================================== B. TRAIN MODEL 2 ================================================== #

# New Model Name
# Build the model
# Initializing cross-validation with 5 folds
i = 1
kf = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)
for train_index,test_index in kf.split(X_train,y):
    print('\n{} of kf {}'.format(i,kf.n_splits))
    xtr,xvl= X_train.loc[train_index],X_train.loc[test_index]
    ytr,yvl= y_train.loc[train_index],y_train.loc[test_index]

    # Initializing logistic regression model
    modelLog = LogisticRegression(random_state=42)

    # Training the model
    modelLog.fit(xtr,ytr)

    # Predicting on validation data
    pred_test= modelLog.predict(xvl)

    # Calculating accuracy score on validation data
    score =accuracy_score(yvl,pred_test)
    print('accuracy_score',score)
    i+= 1

# Predicting on test data
pred_test= modelLog.predict(X_test)

# Storing probability prediction for validation data
pred = modelLog.predict_proba(xvl)[:,1]


# Evaluate the model on the testing data


"""
#Answers: 


#Insight Gained:
    - 
"""
# Create dummy feature importance values
feature_names = X_train.columns  
importance_values = [0.1] * len(feature_names)  # Example: assigning same importance value for all features please fix it with our values

# Save feature importance values to CSV file
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_values})
feature_importance_df.to_csv('feature_importance.csv', index=False)

# Create submission DataFrame and save to CSV file
submission = pd.DataFrame({'Loan_ID': range(1, len(X_test) + 1), 'Loan_status': pred_test})
submission['Loan_status'].replace(0, 'N', inplace=True)
submission['Loan_status'].replace(1, 'Y', inplace=True)
submission.to_csv('ReggModel2log.csv', index=False)



# -------------------------------------------------------
# New Model Name
# Build the model
i = 1
kf = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)
for train_index,test_index in kf.split(X_train,y):
    print('\n{} of kf {}'.format(i,kf.n_splits))
    xtr,xvl= X_train.loc[train_index],X_train.loc[test_index]
    ytr,yvl= y_train.loc[train_index],y_train.loc[test_index]

    # Initializing logistic regression model
    modelTree = tree.DecisionTreeClassifier(random_state=42)

    # Training the model
    modelTree.fit(xtr,ytr)

    # Predicting on validation data
    pred_test= modelTree.predict(xvl)

    # Calculating accuracy score on validation data
    score =accuracy_score(yvl,pred_test)
    print('accuracy_score',score)
    i+= 1

# Predicting on test data
pred_test= modelTree.predict(X_test)

# Storing probability prediction for validation data
pred = modelTree.predict_proba(xvl)[:,1]

"""
Answers:

"""

# Hyperparameters the  model ....
param_grid = {
    'max_depth': list(range(1, 20, 2)),
    'min_samples_split': list(range(2, 20, 2)),
    'min_samples_leaf': list(range(1, 20, 2))
}

# Rebuild the Model

grid_search = GridSearchCV(estimator=modelTree, param_grid=param_grid, cv=kf)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Print the best hyperparameters found
print("Best hyperparameters:", grid_search.best_params_)

# Predicting on test data using the best model
pred_test = best_model.predict(X_test)

# Storing probability prediction for validation data 
pred = best_model.predict_proba(xvl)[:, 1]

# Calculate the mean validation accuracy score
mean_validation_accuracy = grid_search.best_score_
print("Mean validation accuracy:", mean_validation_accuracy)



"""
Answer:


    Mean validation accuracy score: 
"""


# Preprocess the test data in the same way as the training data
test_processed = pd.get_dummies(y_test)

# Make sure the processed test data has the same columns as the training data
missing_columns = set(X_train.columns) - set(test_processed.columns)
for column in missing_columns:
    test_processed[column] = 0

# Reorder columns to match the order of columns in X_train
test_processed = test_processed[X_train.columns]

# Predicting on test data using the best model
pred_test = best_model.predict(test_processed)

# Calculate the mean validation accuracy score
mean_validation_accuracy = grid_search.best_score_
print("Mean validation accuracy:", mean_validation_accuracy)

# Save predictions to a CSV file
submission = pd.DataFrame({'Loan_ID': y_test['Loan_ID'], 'Loan_Status': pred_test})
submission.to_csv('test_predictions.csv', index=False)
# Create submission DataFrame and save to CSV file
submission = pd.DataFrame({'Prediction_ID': range(1, len(X_test) + 1), 'Loan_Status': pred_test})
submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace=True)
submission.to_csv('DecisModel2Log.csv', index=False)

"""
#Answer: Mean validation accuracy score: 

Insight Gained:
    - 
"""

# -------------------------------------------------------
# New Model Name
# Build the model
i = 1
kf = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)
for train_index,test_index in kf.split(X_train,y):
    print('\n{} of kf {}'.format(i,kf.n_splits))
    xtr,xvl= X_train.loc[train_index],X_train.loc[test_index]
    ytr,yvl= y_train.loc[train_index],y_train.loc[test_index]

    # Initializing logistic regression model
    modelFor = RandomForestClassifier(random_state=42 , max_depth=3, n_estimators = 41)

    # Training the model
    modelFor.fit(xtr,ytr)

    # Predicting on validation data
    pred_test= modelFor.predict(xvl)

    # Calculating accuracy score on validation data
    score =accuracy_score(yvl,pred_test)
    print('accuracy_score',score)
    i+= 1

# Predicting on test data
pred_test= modelLog.predict(X_test)

# Storing probability prediction for validation data
pred = modelLog.predict_proba(xvl)[:,1]

# Calculate the mean validation accuracy score
mean_validation_accuracy = accuracy_score(yvl, pred_test)
print("Mean validation accuracy:", mean_validation_accuracy)

"""
Answers:


    Mean validation accuracy score: 
"""


# Hyperparameter Tuning for .....
param_gridFor = {
    'max_depth': list(range(1, 20, 2)),
    'n_estimators': list(range(1, 200, 20))
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=param_gridFor, cv=kf)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Predict on test data using the best model
pred_test = best_model.predict(X_test)
# Rebuild the Model


# Calculate the mean validation accuracy score
mean_validation_accuracy = grid_search.best_score_
print("Mean validation accuracy:", mean_validation_accuracy)


"""
Answer:


    Mean validation accuracy score: 
"""


# Preprocess the test data in the same way as the training data
test_processed = pd.get_dummies(y_test)
missing_columns = set(X_train.columns) - set(test_processed.columns)
for column in missing_columns:
    test_processed[column] = 0
test_processed = test_processed[X_train.columns]



# Reorder columns to match the order of columns in X_train
test_processed = test_processed[X_train.columns]  

# Now you can make predictions on the processed test data
pred_test = best_model.predict(test_processed)

# Create submission DataFrame and save to CSV file
submission = pd.DataFrame({'Prediction_ID': range(1, len(test_processed) + 1), 'Loan_Status': pred_test})
submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace=True)
submission.to_csv('test_predictions.csv', index=False)
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
