# %% [markdown]
# # PYTHON LIBRARIES

# %% [markdown]
# LOADING PYTHON LIBRARIES

# %%
# dash Libraries
import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output

# plotly Libraries
import plotly.graph_objects as go

# pandas Libraries
import pandas as pd


# keras Library
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# numpy Library
import numpy as np

# %% [markdown]
# DATA CLEANING AND SCALING

# %%
# dataset reading
df_nse = pd.read_csv("data/TATACONSUM.NS_historical_data.csv")
df = pd.read_csv("data/stock_data.csv")

# %%
# index formatting
df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
df_nse.index=df_nse['Date']
df_nse = df_nse.sort_index(ascending=False, axis=0)

# %%
# data sorting
data=df_nse.sort_index(ascending=True,axis=0)
new_data=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close'])

# %%
for i in range(0,len(data)):
    new_data["Date"][i]=data['Date'][i]
    new_data["Close"][i]=data["Close"][i]

new_data.index=new_data.Date
new_data.drop("Date",axis=1,inplace=True)

dataset=new_data.values

# %% [markdown]
# Dataset split to Train and Test

# %%
total_sample = dataset.shape[0]
train_sample = int(0.8 * total_sample)

# %%
train=dataset[0:train_sample,:]
valid=dataset[train_sample:,:]

# %% [markdown]
# Scaling of data

# %%
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

# %%
x_train, y_train = [],[]

for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])

# %%
x_train,y_train=np.array(x_train),np.array(y_train)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

# %% [markdown]
# MODEL LOADING

# %%
model = load_model("data/saved_lstm_model.h5")

# %%
inputs=new_data[len(new_data)-len(valid)-60:].values
inputs=inputs.reshape(-1,1)
inputs=scaler.transform(inputs)

# %%
X_test=[]
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)

# %% [markdown]
# DATASET SPLIT TO VALIDATION AND TRAINING

# %%
train=new_data[:train_sample]
valid=new_data[train_sample:]
# valid['Predictions']=closing_price

# %%
valid.loc[:,['Predictions']] = closing_price

# %% [markdown]
# REGREESION METRICS FOR MODEL

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

# %%
mae = mean_absolute_error(valid["Close"], valid["Predictions"])
print("mean_absolute_error",mae)


mse = mean_squared_error(valid["Close"], valid["Predictions"])
print("mean_squared_error",mse)


rmse = np.sqrt(mean_squared_error(valid["Close"], valid["Predictions"]))
print("root_mean_squared_error",rmse)

r2 = r2_score(valid["Close"], valid["Predictions"])
print("r-squared_score",r2)

# %%

recent_data = valid.tail(4)

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
								x=train.index,
								y=train["Close"],
								mode='lines'
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
									x=valid.index,
									y=valid["Predictions"],
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
            go.Scatter(x=valid.index,
                       y=valid[stock],
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


