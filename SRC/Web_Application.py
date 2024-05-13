import dash
from dash import html, dcc
import plotly.graph_objs as go
from dash.dependencies import Input, Output,State
import pickle
import numpy as np
import pandas as pd

with open('Artifacts/Models/Model_2.pkl', 'rb') as file:
    model = pickle.load(file)

# Define feature_names as a global variable
try:
    feature_names = model.feature_names_in_
except AttributeError:
    feature_names = ['loan_int_rate', 'income_to_age_ratio', 'loan_amt_to_income_ratio', 'loan_percent_income', 
                        'emp_length_to_age_ratio', 'person_home_ownership_RENT', 'person_cred_hist_length', 
                        'loan_intent_HOMEIMPROVEMENT', 'person_home_ownership_MORTGAGE', 'loan_intent_MEDICAL', 
                        'person_home_ownership_OWN', 'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION', 
                        'loan_intent_VENTURE', 'loan_intent_PERSONAL', 'person_home_ownership_OTHER']

# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1("Credit Risk Assessment Application"),
    html.Div(id='container', children=[
        html.Label("Age:"),
        dcc.Input(id='person_age', type='number', placeholder = 'Age'),
        html.Label("Income"),
        dcc.Input(id = "person_income", type = 'number', placeholder = "Income"),
        html.Label("Home Ownership:"),
        dcc.Dropdown(id="person_home_ownership", options = [
            {'label': 'Own', 'value':'OWN'},
            {'label':'Rent','value':'RENT'},
            {'label':'Mortgage','value': 'MORTGAGE'},
            {'label':'Other', 'value':'OTHER'}
        ], placeholder="Select Your Home Ownership"),
        html.Label("Employment Length in years"),
        dcc.Input(id="person_emp_length",type='number',placeholder='Enter years in employment'),
        html.Label("Loan Intent"),
        dcc.Dropdown(id='loan_intent', options = [
            {'label':'Personal','value':'Personal'},
            {'label':'Education','value':'EDUCATIONAL'},
            {'label':'Medical', 'value':'MEDICAL'},
            {'label':'Venture','value':'VENTURE'},
            {'label':'Home Improvements','value':'HOMEIMPROVEMENTS'},
            {'label':'Debt Consolidation','value':'DEBTCONSOLIDATION'}],
        placeholder = 'Select Loan Intent'),
        html.Label("Loan Amount:"),
        dcc.Input(id = 'loan_amnt', type='number', placeholder='Loan Amount'),
        html.Label("Interest Rate:"),
        dcc.Slider(
            id='loan_int_rate',
            min=0,
            max=20,  # Assuming the interest rate ranges from 0% to 20%
            step=0.1,
            value=5,  # Default value set at 5%
            tooltip={'always_visible': True, 'placement': 'bottom'},  # Tooltip to show the value always
            marks={i: f'{i}%' for i in range(0, 21, 5)}  # Marks at every 5%
        ),
        html.Label("Loan Percent of Income:"),
        dcc.Slider(
            id='loan_percent_income',
            min=0,
            max=1,  
            step=0.01,
            value=0.1,  
            tooltip={'always_visible': True, 'placement': 'bottom'},     marks={i/10: f'{int(i*10)}%' for i in range(0, 11, 2)}  # Marks at every 20%
        ),
        html.Label("Credit History Length in years:"),
        dcc.Input(id='person_cred_hist_length',type='number',placeholder='Credit History length in years' ),
        html.Button("Submit", id = 'submit',n_clicks=0),
        ]),
        html.Div(id='prediction-output', children='Fill in the form and press Enter')
])

@app.callback(
    Output('prediction-output', 'children'),
    [Input('submit', 'n_clicks')],
    [State('person_age', 'value'),
        State('person_income', 'value'),
        State('person_home_ownership', 'value'),
        State('person_emp_length', 'value'),
        State('loan_intent', 'value'),
        State('loan_amnt', 'value'),
        State('loan_int_rate', 'value'),
        State('loan_percent_income', 'value'),
        State('person_cred_hist_length', 'value')])

def update_output(n_clicks, person_age, person_income, person_home_ownership, person_emp_length,
                    loan_intent, loan_amnt, loan_int_rate, loan_percent_income,person_cred_hist_length):
    if n_clicks > 0:
        income_to_age_ratio = person_income / person_age if person_age != 0 else 0
        loan_amt_to_income_ratio = loan_amnt / person_income if person_income != 0 else 0
        emp_length_to_age_ratio = person_emp_length / person_age if person_age != 0 else 0

        # Create a template DataFrame with all the necessary features
        template = pd.DataFrame(columns=feature_names, data=np.zeros((1, len(feature_names))))

        # Update the values based on the user's input
        template.at[0, 'loan_int_rate'] = loan_int_rate
        template.at[0, 'income_to_age_ratio'] = income_to_age_ratio
        template.at[0, 'loan_amt_to_income_ratio'] = loan_amt_to_income_ratio
        template.at[0, 'loan_percent_income'] = loan_percent_income
        template.at[0, 'emp_length_to_age_ratio'] = emp_length_to_age_ratio
        template.at[0, 'person_home_ownership_' + person_home_ownership] = 1
        template.at[0, 'person_cred_hist_length'] = person_cred_hist_length
        template.at[0, 'loan_intent_' + loan_intent] = 1
        # Make prediction
        prediction = model.predict(template)

        # Return the prediction result
        if prediction[0] == 1:
            return html.Div('Approved', style={'color': 'green'})
        else:
            return html.Div('Rejected', style={'color': 'red'})
    else:
        return 'Fill in the form and press Enter'

if __name__ == "__main__":
    app.run_server(debug=True)