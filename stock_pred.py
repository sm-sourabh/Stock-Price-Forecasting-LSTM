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
from keras.callbacks import EarlyStopping
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

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
train_size = int(0.8 * total_sample)
valid_size = int(0.1 * train_size)

# %%
train_data = final_dataset[0:train_size,:]
test_data = final_dataset[train_size-valid_size:train_size, :]
valid_data = final_dataset[train_size:, :]


# %%
scaler=MinMaxScaler(feature_range=(0,1))
scaled_train_data = scaler.fit_transform(train_data)
scaled_valid_data = scaler.transform(test_data)
scaled_test_data = scaler.transform(valid_data)


# %%
window_size = 60

# %%[markdown]
# Creating NP-Arrys for LSTM Model
# %%
x_train_data,y_train_data=[],[]

for i in range(window_size,len(scaled_train_data)):
    x_train_data.append(scaled_train_data[i-window_size:i,0])
    y_train_data.append(scaled_train_data[i,0])

x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)
x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

# %%
x_valid_data, y_valid_data = [], []

for i in range(window_size, len(scaled_valid_data)):
    x_valid_data.append(scaled_valid_data[i - window_size:i, 0])
    y_valid_data.append(scaled_valid_data[i, 0])

x_valid_data, y_valid_data = np.array(x_valid_data), np.array(y_valid_data)
x_valid_data = np.reshape(x_valid_data, (x_valid_data.shape[0], x_valid_data.shape[1], 1))

# %% [markdown]
# LSTM MODEL BUILDING
# %%
lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))

# model compilation
lstm_model.compile(loss='mean_squared_error',optimizer='adam')

early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=1, restore_best_weights=True)
lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=32, validation_data=(x_valid_data, y_valid_data), callbacks=[early_stop], verbose='verbose=2')

# %%
inputs_data=new_dataset[len(new_dataset)-len(valid_data)-window_size:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)

x_test_data, y_test_data = [], []

for i in range(window_size, inputs_data.shape[0]):
    x_test_data.append(inputs_data[i - window_size:i, 0])
    y_test_data.append(inputs_data[i, 0])

x_test_data, y_test_data = np.array(x_test_data), np.array(y_test_data)
x_test_data = np.reshape(x_test_data, (x_test_data.shape[0], x_test_data.shape[1], 1))


# %% [markdown]
# PREDICTIONS BY LSTM MODEL
# %%

closing_price=lstm_model.predict(x_test_data)
closing_price=scaler.inverse_transform(closing_price)

# %%
# saving the predictions to csv
train_data=new_dataset[:train_size-valid_size]
test_data=new_dataset[train_size-valid_size:train_size]
valid_data=new_dataset[train_size:]
valid_data['Predictions'] = closing_price

valid_data.to_csv("data/lstm_predictions.csv")

# %% [markdown]
# Model Summary

# %%
#model Summary
test_loss = lstm_model.evaluate(x_test_data, y_test_data)
print("Test Loss:", test_loss)
lstm_model.summary()

# REGREESION METRICS FOR MODEL
mae = mean_absolute_error(valid_data["Close"], valid_data["Predictions"])
print("mean_absolute_error",mae)

mse = mean_squared_error(valid_data["Close"], valid_data["Predictions"])
print("mean_squared_error",mse)

rmse = np.sqrt(mean_squared_error(valid_data["Close"], valid_data["Predictions"]))
print("root_mean_squared_error",rmse)

r2 = r2_score(valid_data["Close"], valid_data["Predictions"])
print("r-squared_score",r2)

# saving the outcome in .h5 file
lstm_model.save("data/saved_lstm_model.h5")

# %% [markdown]
## GRAPH PLOTING

# %%
# ploting the outputs

plt.plot(train_data["Close"])
plt.plot(test_data["Close"])
plt.plot(valid_data[['Close',"Predictions"]])

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


plt.figure(figsize=(16,8),dpi=dpi)
plt.plot(train_data.index, train_data["Close"],
         label="Training Data",
         color="Blue")
plt.plot(test_data.index, test_data['Close'],
         label='Test Data',
         color='DarkOrange')
plt.plot(valid_data.index, valid_data['Close'],
         label="Validation Data",
         color="Green")
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


plt.figure(figsize=(16,8),dpi=dpi)

plt.plot(valid_data.index, valid_data["Close"],
         label="Actual Close",
         color="green")
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Actual Close Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
