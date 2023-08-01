import numpy as np
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

class STOCK:
    df1 = None
    df2 = None
    scaler = None
    look_back = 100
    train_predict = None
    test_predict = None
    key = None
    type = None
    train_data = None
    test_data = None
    x_train = None
    y_train = None
    model = None




    def __init__(self, type):
        self.key = "40d8d2da9b03f5b83672c59955b2e875b29eeb2f"
        self.type = type
        self.fetch_data()
        self.showCurrent_data()
        self.create_LSTM_data()
        self.train_test_data()
        self.show_train_test_data()
        self.forecasting()

    def fetch_data(self):
        df = pdr.get_data_tiingo(self.type, api_key=self.key)
        df.to_csv(self.type + '.csv')
        self.df1 = df.reset_index()['close']
        self.df2 = df.reset_index()['date']

    def showCurrent_data(self):
        fig, ax = plt.subplots()
        ax.plot(self.df2, self.df1)
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.set_title('Stock Data')
        path = 'D:/Stock_Prediction_System/static/' + self.type + '_PLOTTED_DATA.png'
        ax.figure.savefig(path)
        plt.close(ax.figure)

    def create_dataset(self, dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    def create_LSTM_data(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.df1 = self.scaler.fit_transform(np.array(self.df1).reshape(-1, 1))
        train_size = int(len(self.df1) * 0.65)
        test_size = len(self.df1) - train_size
        self.train_data, self.test_data = self.df1[0:train_size, :], self.df1[train_size:len(self.df1), :]
        time_step = 100 
        self.x_train, self.y_train = self.create_dataset(self.train_data, time_step)
        self.x_test, self.y_test = self.create_dataset(self.test_data, time_step)
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1)

    def train_test_data(self):
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
        self.model.add(LSTM(50, return_sequences=True))
        self.model.add(LSTM(50))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=50, batch_size=64, verbose=1)
        self.train_predict = self.model.predict(self.x_train)
        self.test_predict = self.model.predict(self.x_test)
        self.train_predict = self.scaler.inverse_transform(self.train_predict)
        self.test_predict = self.scaler.inverse_transform(self.test_predict)

    def show_train_test_data(self):
        trainPredictPlot = np.empty_like(self.df1)
        trainPredictPlot[:, :] = np.nan 
        trainPredictPlot[self.look_back:len(self.train_predict)+self.look_back, :] = self.train_predict

        testPredictPlot = np.empty_like(self.df1) 
        testPredictPlot[:, :] = np.nan 
        testPredictPlot[len(self.train_predict)+self.look_back+101:len(self.df1)-1, :] = self.test_predict

        fig, ax = plt.subplots()
        ax.plot(self.df2,self.scaler.inverse_transform(self.df1))
        ax.plot(self.df2,trainPredictPlot)
        ax.plot(self.df2,testPredictPlot)
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.set_title('Stock Prediction and Testing')
        ax.legend(['Original Data', 'Training Prediction', 'Test Prediction'])
        path = 'D:/Stock_Prediction_System/static/' + self.type + '_PREDICTED_DATA.png'
        ax.figure.savefig(path)
        plt.close(ax.figure)

    def forecasting(self):
        x_input=self.test_data[341:].reshape(1,-1)
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()
        lst_output=[]
        n_steps=100
        i=0
        while(i<10):
            if(len(temp_input)>100):
                x_input=np.array(temp_input[1:])
                x_input=x_input.reshape(1,-1)
                x_input=x_input.reshape(1,n_steps,1)
                yhat = self.model.predict(x_input,verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                lst_output.extend(yhat.tolist())
                i = i+1
            else:
                x_input=x_input.reshape((1,n_steps-1,1))
                yhat = self.model.predict(x_input,verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i = i+1
        
        last_date = self.df2[len(self.df2)-1]
        additional_dates = []
        for i in range(1, 11):
            next_date = last_date + timedelta(days=i)
            additional_dates.append(next_date)
        df4 = self.df2[-100:]

        fig, ax = plt.subplots()
        ax.plot(df4,self.scaler.inverse_transform(self.df1[1156:]))
        ax.plot(additional_dates,self.scaler.inverse_transform(lst_output))
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.set_title('10 days Forecasting')
        ax.legend(['Original Data', 'Future Prediction'])
        path = 'D:/Stock_Prediction_System/static/' + self.type + '_FORECASTED_DATA.png'
        ax.figure.savefig(path)
        plt.close(ax.figure)
        
