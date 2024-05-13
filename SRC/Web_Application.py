import dash
from dash import html, dcc
import plotly.graph_objs as go
from dash.dependencies import Input, Output,State
import pickle
import numpy as np

'''model_path = 'model.pk1'
with open(model_path, 'rb') as file:
    model = pickle.load(file)'''
    
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Credit Risk Assessment Application"),
    html.Div([
        html.Label("Age:"),
        dcc.Input(id='person_age', type='number', placeholder = 'Age'),
        html.Label("Income"),
        dcc.Input(id = "person_income", type = 'number', placeholder = "Income"),
        html.Label("Home Ownership:"),
        dcc.Dropdown(id="person_home_ownership", options = [
            {'label': 'Own', 'value':'OWN'},
            {'label':'Rent','value':'RENT'},
            {'label':'Morgage','value': 'MORGAGE'},
            {'label':'Other', 'value':'OTHER'}
        ], placeholder="Select Your Home Ownership"),
        html.Label("Employment Length in years"),
        dcc.Input(id="person_emp_length",type='number',placeholder='Enter years in employment'),
        html.Label("Loan Intenet"),
        dcc.Dropdown(id='loan_intent', options = [
            {'label':'Personal','value':'Personal'},
            {'label':'Education','value':'EDUCATIONAL'},
            {'label':'Medical', 'value':'MEDICAL'},
            {'label':'Venture','value':'VENTURE'},
            {'label':'Home Improvements','value':'HOMEIMPROVEMENTS'},
            {'label':'Debt Consolidation','value':'DEBTCONSOLIDATION'}],
        placeholder = 'Select Loan Intenet'),
        html.Label("Loan Amount:"),
        dcc.Input(id = 'loan_amnt', type='number', placeholder='Loan Amount'),
        html.Label("Interest Rate:"),
        dcc.Input(id='loan_int_rate', type='number', placeholder = 'Interest Rate',step=0.01),
        html.Label("Loan Percent Income:"),
        dcc.Input(id='loan_percent_income', type='number',placeholder = 'Loan percent Income',step=0.01),
        html.Label("Credit Default on File:"),
        dcc.Dropdown(id='cb_person_default_on_file',options=
                    [
                    {'label':'Yes','value':'Y'},
                    {'label':'No','value':'N'}
                    ],placeholder = "Credit default on file?"),
        html.Label("Credit History Length in years:"),
        dcc.Input(id='cb_person_cred_hist_length',type='number',placeholder='Credit History length in years' ),
        html.Button("Submit", id = 'submit',n_clicks=0),
        ]),
        html.Div(id='prediction-output', children='Fill in the form and press Enter')    
        
    ])

@app.callback(
    Output('prediction-output','children'),
    [Input('submit','n_clicks')],
    [State('person_age','value'),
    State('person_income','value'),
    State('person_home_ownership','value'),
    State('person_emp_length','value'),
    State('loan_intent','value'),
    State('loan_amnt','value'),
    State('loan_int_rate','value'),
    State('loan_percent_income','value'),
    State('cb_person_default_on_file','value'),
    State('cb_person_cred_hist_length','value')])

def update_output(n_clicks, person_age, person_income, person_home_ownership, person_emp_length,
                loan_intent, loan_amnt, loan_int_rate, loan_percent_income, cb_person_default_on_file,
                cb_person_cred_hist_length):
    if n_clicks > 0:
        #prediction goes here

    
    
        return "Simulated Prediction: 0 "
    else:
        return 'Fill in the form and press Enter'
    
if __name__ == "__main__":
    app.run_server(debug=True)
        
        
    
'''input_data = np.array([[person_age, person_income, person_home_ownership, person_emp_length,
                loan_intent, loan_amnt, loan_int_rate, loan_percent_income, cb_person_default_on_file,
                cb_person_cred_hist_length]])
        
        prediction = model.predict(input_data)'''
