from matplotlib.pyplot import hist
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import math 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

class WorkloadArima:
    def __init__(self,path, train_rate = 0.8,tdata = 1):
        self.path = path
        self.train_rate = train_rate

        self.tdata = tdata
        self.tmodel = 'arima'

        self.data = pd.read_csv(self.path)
        self.X = self.data.values
    
        length_split = int(len(self.data)* train_rate)
        self.train_data = self.data[:length_split]
        self.test_data = self.data[length_split:]

    def run_predict(self):
        size = int(len(self.X) * 0.8)
        train, test = self.X[0:size], self.X[size:len(self.X)]
        history = [x[1] for x in train]
        test_val = [x[1] for x in test]
        predictions = list()
        for t in range(len(test_val)):
            model = ARIMA(history, order=(5,1,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t][1]
            history.append(obs)
            #print('predicted=%f, expected=%f' % (yhat, obs))
        # compare the 25% tested values with predicted values

        mae = mean_absolute_error(test_val, predictions)
        rmse = math.sqrt(mean_squared_error(test_val, predictions))
        print('MAE: %.3f' % mae)
        print('RMSE: %.3f' % rmse)
        
        #export data
        self.test_data['y_pred'] = predictions
        dpath = "estimator/ts{}_test_{}.csv".format(self.tdata, self.tmodel)
        self.test_data.to_csv(dpath,index=None)
        print(dpath)
        
        plt.plot(test_val)
        plt.plot(predictions, color='red')
        plt.show()        

x = WorkloadArima('estimator/data_est/nasa_train_test.csv',tdata=2)
x.run_predict()

# data = pd.read_csv('estimator\clarknet_train_test.csv')
# X = data.values

# size = int(len(X) * 0.8)
# train, test = X[0:size], X[size:len(X)]

# history = [x[1] for x in train]
# test_val = [x[1] for x in test]
# predictions = list()
# for t in range(len(test)):
#     model = ARIMA(history, order=(5,1,0))
#     model_fit = model.fit()
#     output = model_fit.forecast()
#     yhat = output[0]
#     predictions.append(yhat)
#     obs = test[t][1]
#     history.append(obs)
#     print('predicted=%f, expected=%f' % (yhat, obs))
# # compare the 25% tested values with predicted values
# plt.plot(test_val)
# plt.plot(predictions, color='red')
# plt.show()