import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.recurrent import LSTM
from datetime import date
import gradio as gr

def create_dataset(dataset,time_step=15):
    x_ind,y_dep =[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        x_ind.append(a)
        y_dep.append(dataset[i+time_step,0])
    return np.array(x_ind),np.array(y_dep)

def stockprice(stockname,number_of_samples):
    df_yahoo = yf.download(stockname,start='2020-09-15',end=date.today(),interval = "1h",progress=False,auto_adjust=True)
    df=df_yahoo
    df.index.rename('Date', inplace=True)
    df=df.sort_values(by=['Date'],ignore_index=True)
    min_max_scaler=MinMaxScaler(feature_range=(0,1))
    dataset=min_max_scaler.fit_transform(df['Close'].values.reshape(-1,1))
    train_size=int(len(df)*0.8)
    test_size=len(df)-train_size
    Train=dataset[0:train_size,:]
    Test=dataset[train_size:len(dataset),:]
    x_train,y_train=create_dataset(Train,time_step=15)
    x_test,y_test=create_dataset(Test,time_step=15)
    x_train=np.reshape(x_train,(x_train.shape[0],1,x_train.shape[1]))
    x_test=np.reshape(x_test,(x_test.shape[0],1,x_test.shape[1]))
    time_step=15
    model=Sequential()
    model.add(LSTM(20,input_shape=(1,time_step)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",optimizer='adam')
    model.fit(x_train,y_train,epochs=100,verbose=0)
    y_pred=model.predict(x_test)
    y_pred_RNN=min_max_scaler.inverse_transform(y_pred)
    y_test=np.expand_dims(y_test,axis=1)
    y_test=min_max_scaler.inverse_transform(y_test)
    df1=df.drop(["Volume","Open","High","Low"],axis=1)
    a= int(number_of_samples)*15
    new_data = df1[-(a+1):-1]
    last60prices=np.array(new_data)
    last60prices=last60prices.reshape(-1, 1)
    X=min_max_scaler.transform(last60prices)
    TimeSteps=int(15)
    NumFeatures=int(1)
    number_of_samples=int(number_of_samples)
    X=X.reshape(number_of_samples, NumFeatures, TimeSteps)
    predicted_Price = model.predict(X)
    predicted_Price = min_max_scaler.inverse_transform(predicted_Price)
    pred_df=pd.DataFrame(list(map(lambda x: x[0], predicted_Price)),columns=["PREDICTIONS"])

    pred_df.reset_index(inplace=True)
    pred_df = pred_df.rename(columns = {'index':'HOURS'})
    
    plt.figure(figsize=(15, 6))
    range_history = len(new_data)
    range_future = list(range(range_history, range_history +len(predicted_Price)))
    plt.plot(np.arange(range_history), np.array(new_data),label='History')
    plt.plot(range_future, np.array(predicted_Price),label='Forecasted for RNN')
    plt.legend(loc='upper right')
    plt.xlabel('Time step (hour)')
    plt.ylabel('Stock Price')
    
    
    return pred_df,plt.gcf()

interface = gr.Interface(fn = stockprice, 
inputs = [gr.inputs.Textbox(lines=1, placeholder="Enter STOCK-TICKER", default="FB", label="STOCKNAME"),
gr.inputs.Slider(minimum=0, maximum=150, step=1, default=5, label="Number of Sample to Predict")], 
outputs = ["dataframe","plot"],
description="LSTM STOCK PREDICTION")

interface.launch()