# %% [markdown]
## PYTHON LIBRARIES

# %% [markdown]
# LOADING PYTHON LIBRARIES
# %%
# libraries
import pandas as pd
import numpy as np
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler

# %% [markdown]
# DATA EXTRACTION
# %%
def extract_data(ticker_symbol,csv_location="data/"):
    stock_data = yf.download(ticker_symbol, period='max')
    df_stock = pd.DataFrame(stock_data)
    csv_filename = f"{ticker_symbol}_historical_data.csv"
    csv = csv_location + csv_filename
    df_stock.to_csv(csv)
    print(f"Data saved to {csv}")
    # return df_stock  # Optionally, return the DataFrame

if __name__ == "__main__":
    ticker_symbol = "TATACONSUM.NS"
    extract_data(ticker_symbol)


# %%
rcParams['figure.figsize']=20,10

# %% [markdown]
# DATA SCALING
# %%
scaler=MinMaxScaler(feature_range=(0,1))

# %% [markdown]
# LOADING THE CSV FILE
# %%
df=pd.read_csv("data/TATACONSUM.NS_historical_data.csv")

# %%
# ploting the data on a line graph
sns.set(rc={'figure.figsize' :(20,5)})
df['Open'].plot(linewidth=1,color='blue')

# %%
# chaning the index to Date
df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df["Date"]


# %% [markdown]
# PLOTING ALL THE VARIABLES
# %%
cols_plot=['Open','High','Low','Close','Adj Close']
colors = ['blue', 'green', 'red', 'purple', 'orange']
dpi = 300
fig,axes=plt.subplots(nrows=len(cols_plot),figsize=(16,7*len(cols_plot)), dpi=dpi)
for i, col in enumerate(cols_plot):
    df[col].plot(ax=axes[i], alpha=1, color=colors[i], label=col)
    axes[i].set_ylabel(col + ' Variation')
    axes[i].set_xlabel('Date')
    axes[i].grid(True)
    axes[i].legend()
fig.suptitle('Stock Price Variations', fontsize=16)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(16,8))
plt.plot(df["Close"],label='Close Price history')


# %%
# data sorting 
data=df.sort_index(ascending=True,axis=0)
new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])

new_dataset = data[['Date', 'Close']].copy()

new_dataset.index=new_dataset.Date

new_dataset.drop("Date",axis=1,inplace=True)

final_dataset=new_dataset.values

# %% [markdown]
# DATA SPLIT INTO TEST AND TRAIN
# %%
total_sample = final_dataset.shape[0]
train_sample = int(0.7 * total_sample)

# %%
train_data=final_dataset[0:train_sample,:]
valid_data=final_dataset[train_sample:,:]

# %%
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_dataset)

# %%
x_train_data,y_train_data=[],[]

# %%
scaled_data[0:60,0]
scaled_data[120,0]

# %%
for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])

# %%
x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)
x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

# %% [markdown]
# LSTM MODEL BUILDING
# %%
lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

# %%
lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose='verbose=2')

# %%
lstm_model.summary()

# %%
inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)

# %%
X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)

# %%
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=lstm_model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)

# %%
# saving the outcome in .h5 file
lstm_model.save("data/saved_lstm_model.h5")

# %%
# ploting the outputs
train_data=new_dataset[:train_sample]
valid_data=new_dataset[train_sample:]
valid_data['Predictions']=closing_price
plt.plot(train_data["Close"])
plt.plot(valid_data[['Close',"Predictions"]])

# %% [markdown]
## GRAPH PLOTING
# %%
dpi = 600
plt.figure(figsize=(16,8),dpi=dpi)

plt.plot(train_data.index, train_data["Close"],
         label="Training Data",
         color="#0055b2")
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Training Stock Data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%

dpi = 600
plt.figure(figsize=(16,8),dpi=dpi)
plt.plot(valid_data.index, valid_data['Close'],
         label="Validation Data",
         color="green")
plt.plot(valid_data.index, valid_data["Predictions"],
         label="Predictions",
         color="red")
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Comparision of Actual and Predicted Values')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
dpi = 600
plt.figure(figsize=(16,8),dpi=dpi)
plt.plot(train_data.index, train_data["Close"],
         label="Training Data",
         color="#0055b2")
plt.plot(valid_data.index, valid_data['Close'],
         label="Validation Data",
         color="green")
plt.plot(valid_data.index, valid_data["Predictions"],
         label="Predictions",
         color="red")
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
