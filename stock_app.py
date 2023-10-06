# %% [markdown]
##  LOADING stock_pred.py
# stock_pred.py

# %%
import stock_pred

# %% [markdown]
## LOADING stock_app.py
# stock_app.py
# %% [markdown]
# LOADING PYTHON LIBRARIES

# %%
# python Libraries
import pandas as pd
import numpy as np


# dash Libraries
import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output

# plotly Libraries
import plotly.graph_objects as go



# %% [markdown]
# DATA CLEANING AND SCALING

# %%
# dataset reading
df_nse = pd.read_csv("data/TATACONSUM.NS_historical_data.csv")
df = pd.read_csv("data/stock_data.csv")
df_pred = pd.read_csv("data/lstm_predictions.csv")

# %%
# index formatting
df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
df_nse.index=df_nse['Date']
df_nse = df_nse.sort_index(ascending=False, axis=0)
df_nse = df_nse.drop('Date', axis=1)

df_pred["Date"]=pd.to_datetime(df_pred.Date,format="%Y-%m-%d")
df_pred.index = df_pred['Date']
df_pred = df_pred.sort_index(ascending=False, axis=0)
df_pred = df_pred.drop('Date', axis=1)

# %% [markdown]
# REGREESION METRICS FOR MODEL

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

# %%
mae = mean_absolute_error(df_pred["Close"], df_pred["Predictions"])
print("mean_absolute_error",mae)


mse = mean_squared_error(df_pred["Close"], df_pred["Predictions"])
print("mean_squared_error",mse)


rmse = np.sqrt(mean_squared_error(df_pred["Close"], df_pred["Predictions"]))
print("root_mean_squared_error",rmse)

r2 = r2_score(df_pred["Close"], df_pred["Predictions"])
print("r-squared_score",r2)

# %%

recent_data = df_pred.head(4)

recent_data['Close'] = pd.to_numeric(recent_data['Close'], errors='coerce')
recent_data['Predictions'] = pd.to_numeric(recent_data['Predictions'], errors='coerce')

recent_data.loc[:, 'Close'] = recent_data['Close'].apply(lambda x: round(x,2))
recent_data.loc[:, 'Predictions'] = recent_data['Predictions'].apply(lambda x: round(x,2))

recent_data['Percentage Change'] = ((recent_data['Predictions'] - recent_data['Close']) / recent_data['Close']) * 100
recent_data['Percentage Change'] = recent_data['Percentage Change'].apply(lambda x: round(x,2))
recent_data['Date'] = recent_data.index

# %%
regression_metrics = {
    'MAE': mae,
    'MSE': mse,
    'RMSE': rmse,
    'R2': r2
}

# %% [markdown]
# # DASHBOARD

# %% [markdown]
# APP SERVER

# %%
app = dash.Dash(__name__)
server = app.server

# %% [markdown]
# APP LAYOUT

# %%
app.layout = html.Div([
   
    html.H1("Stock Price Analysis Dashboard"),
   
    dcc.Tabs(id="tabs", children=[
		# tab1
		dcc.Tab(label="Model and TATA Stock Analysis",className='custom-tab',children=[
            html.Div([
                html.H2("Actual closing price", className='custom-content'),
				dcc.Graph(
					id="Actual Data",
					figure={
						"data":[
							go.Scatter(
								x=stock_pred.train_data.index,
								y=stock_pred.train_data["Close"],
								mode='lines',
                                name='Training Data'
							),
                            go.Scatter(
                                x=stock_pred.test_data.index,
                                y=stock_pred.test_data["Close"],
                                mode='lines',
                                name='Validation Data'
                            )

						],
						"layout":go.Layout(
							title='Actual Closing Rate(1996-2018)',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}
				),
			]),
            
			# 2 panes 
			html.Div([
                # left pane
				html.Div([
            	    html.H2("Model Metrics Table", className='custom-content'),
					dash_table.DataTable(
        				id='metrics-table',
        				columns=[
        			    	{'name': 'Metric', 'id': 'Metric'},
        			    	{'name': 'Value', 'id': 'Value'}
        				],
        				data=[{'Metric': metric, 'Value': value} for metric, value in regression_metrics.items()],
                	),
                    html.H2("Last Updates", className='custom-content'),
					dash_table.DataTable(
        				id='update-table',
        				columns=[
                            {'name': 'Date', 'id': 'Date'},
                            {'name': 'Close', 'id': 'Close'},
                            {'name': 'Predicted Close', 'id': 'Predictions'}, 
                            {'name': 'Percentage Change', 'id': 'Percentage Change'}, 
                        ],
        				data=recent_data.to_dict('records'),
                	),
				], className='pane pane1' ),
            
				# right pane
            	html.Div([
					html.H2("LSTM Predicted closing price", className='custom-content'),
					dcc.Graph(
						id="Predicted-data",
						figure={
							"data":[
								go.Scatter(
									x=df_pred.index,
									y=df_pred["Predictions"],
									mode='lines'
								)
							],
							"layout":go.Layout(
								title='Predicted Closing Rate(2018-Present)',
								xaxis={'title':'Date'},
								yaxis={'title':'Closing Rate'}
							)
						}
					),
				], className='pane pane2'),
            ],className='pane'),
			
		]),
        
		# tab2
        dcc.Tab(label='NSE-TATAGLOBAL Stock Analysis',className='custom-tab', children=[
            html.Div([
                html.H2("Tata Stock Actual vs Predicted closing price", className='custom-content'),

                dcc.Dropdown(id= 'my-dropdown3',
                             options=[{'label': 'Actual Closing', 'value': 'Close'},
                                      {'label': 'Predicted Closing', 'value': 'Predictions'}],
                            multi=True,value=['Close'], className='dropdown-menu'),
                dcc.Graph(id='closing'),
            ]),
            
			html.Div([
                html.H2("Other Stats", className='custom-content'),
                dcc.Checklist(
        			id='toggle-buttons',
        			options=[
            			{'label': 'Open', 'value': 'Open'},
            			{'label': 'High', 'value': 'High'},
            			{'label': 'Low', 'value': 'Low'},
            			{'label': 'Adj Close', 'value': 'Adj Close'},
        			],
        			value=['Open'],  # Default value is an empty list
                    className='dash-checklist'
    			),
                dcc.Graph(id='toggle-button-output'),			
			]),
            
			html.Div([
                html.H2("TATA GLOBAL stock Volume", className='custom-content'),
                dcc.Graph(
					id="Predicted Data",
					figure={
						"data":[
							go.Scatter(
								x=df_nse.index,
								y=df_nse["Volume"],
								mode='lines'
							)
						],
						"layout":go.Layout(
							title='Trade Quantity for Tata Global Stock over Time',
							xaxis={'title':'Date'},
							yaxis={'title':'Transaction Volume'}
						)
					}
				),
			])
        ]),

		# tab3
        dcc.Tab(label='Stock Analysis (AAPL, TSLA, etc)',className='custom-tab', children=[
            html.Div([
                html.H2("Stocks High vs Lows", className='custom-content'),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'}, 
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['FB'], className='dropdown-menu'),
                dcc.Graph(id='highlow'),


                html.H2("Stocks Market Volume", className='custom-content'),
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['FB'],className='dropdown-menu'),
                dcc.Graph(id='volume')
            ]),
        ])


    ])
])

# %% [markdown]
# APP CALLBACKS

# %%
@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph_highlow(selected_dropdown):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure

# %%
@app.callback(Output('closing','figure'),
              [Input('my-dropdown3', 'value')])
def update_graph_closing(selected_dropdown_close):
    dropdown= {"Close": "Actual","Predictions": "Predicted"}
    trace1=[]
    for stock in selected_dropdown_close:
        trace1.append(
            go.Scatter(x=df_pred.index,
                       y=df_pred[stock],
                       mode = 'lines', opacity=0.7,
                       name=f'{dropdown[stock]} Closing ', textposition='bottom center'))
        
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"{''.join(str(dropdown[i]) for i in selected_dropdown_close)} Prices Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Cosing Price"})}
    return figure

# %%
@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph_volume(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Volume"],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure

# %%
@app.callback(Output('toggle-button-output', 'figure'),
              [Input('toggle-buttons', 'value')])
def update_output(selected_values):
    dropdown = {"Open": "Open","High": "High","Low": "Low","Adj Close": "Adj Close",}
    trace1 = []
    for stock in selected_values:
        trace1.append(
          go.Scatter(x=df_nse.index,
                       y=df_nse[stock],
                       mode = 'lines', opacity=0.7,
                       name=f'{dropdown[stock]} ', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"{', '.join(str(dropdown[i]) for i in selected_values)} values Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Prices"})}
    return figure

# %% [markdown]
# APP SERVER START

# %%
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)



# %%
