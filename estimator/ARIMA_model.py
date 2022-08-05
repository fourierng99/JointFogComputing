from matplotlib.pyplot import hist
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import sklearn
rcParams['figure.figsize'] = 15, 6
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import math 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

from statsmodels.graphics.tsaplots import plot_acf
class WorkloadArima:
    def __init__(self, train_rate = 0.8,tdata = 1):

        self.path = 'estimator/data_est/clarknet_train_test.csv'
        if(tdata == 2):
            self.path='estimator/data_est/nasa_train_test.csv'
        self.train_rate = train_rate

        self.tdata = tdata
        self.tmodel = 'arima'


        print(self.path)
        self.data = pd.read_csv(self.path)
        self.X = self.data.values
    
        length_split = int(len(self.data)* train_rate)
        self.train_data = self.data[:length_split]
        self.test_data = self.data[length_split:]

        
        # f,ax = plt.subplots(1,2)
        # ax[0,0].plot(self.train_data['y'].values.diff())
        # plt.show()

        #plot_acf(self.train_data['y'].values)

    def evaluate_arima(self,p, d,q):
        size = int(len(self.X) * self.train_rate)
        train, test = self.X[0:size], self.X[size:]
        history = [x[1] for x in train]
        test_val = [x[1] for x in test]
        predictions = list()
        for t in range(len(test_val)):
            model = ARIMA(history, order=(p,d,q))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t][1]
            history.append(obs)

        #pr = predictions[-1]
        #predictions.append(pr)
        mae = mean_absolute_error(test_val, predictions)
        mape = mean_absolute_percentage_error(y_true=test_val, y_pred=predictions)
        rmse = math.sqrt(mean_squared_error(test_val, predictions))
        print('order = ({0},{1},{2}),RMSE = {3}, MAPE = {4}, MAE = {5}'.format(p,d,q,rmse,mape,mae))   
        self.res_file.write("{},{},{},{},{}\n".format(p,d,q,rmse,mape))  

    def evaluate(self):
        self.res_file = open("estimator/tune_arima_{}.csv".format(self.tdata), 'w')
        self.res_file.write("p,d,q,rmse\n")
        p = [0,1,2,3]
        d = [0,1,2]
        q = [0,1,2]
        for i in p:
            for j in d:
                for k in q:
                    try:
                        self.evaluate_arima(i,j,k) 
                    except:
                        self.res_file.write("{},{},{},error,error\n".format(i,j,k))
        self.res_file.close()

    def run_predict(self):
        size = int(len(self.X) * self.train_rate)
        train, test = self.X[0:size], self.X[size:len(self.X)]
        history = [x[1] for x in train]
        test_val = [x[1] for x in test]
        predictions = list()
        for t in range(len(test_val)):
            model = ARIMA(history, order=(1,2,2))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t][1]
            history.append(obs)
            #print('predicted=%f, expected=%f' % (yhat, obs))
        # compare the 25% tested values with predicted values
        pr = predictions[-1]
        predictions.append(pr)
        mae = mean_absolute_error(test_val, predictions[0:-1])
        mape = mean_absolute_percentage_error(y_true=test_val, y_pred=predictions[0:-1])
        rmse = math.sqrt(mean_squared_error(test_val, predictions[0:-1]))
        print('MAE: %.3f' % mae)
        print('RMSE: %.3f' % rmse)
        print(mape)
        #export data
        self.test_data['y_pred'] = predictions[1:]
        dpath = "estimator/ts{}_test_{}.csv".format(self.tdata, self.tmodel)
        self.test_data.to_csv(dpath,index=None)
        print(dpath)
        
        plt.plot(test_val)
        #plt.plot(predictions[0:], color='red')
        #print(predictions)
        plt.plot(self.test_data['y_pred'].values, color='green')
        plt.plot(xlabel = 'datetime',ylabel ='request count', label='request count')
        plt.show()        

#x = WorkloadArima('estimator/data_est/nasa_train_test.csv',tdata=2)
x = WorkloadArima(tdata=2)
#x.evaluate()
x.run_predict()

