import numpy as np
import pandas as pd
import os 
from prophet.serialize import model_to_json, model_from_json
from prophet import Prophet

from datetime import datetime
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error
import math 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

class WorkloadEstimator:
    def __init__(self,tdata =1):
        self.tdata = tdata
        self.tmodel = 'prophet'
    def build_model(self, model_path = ''):
        if os.path.isfile(model_path):
            with open('serialized_model.json', 'r') as fin:
                self.model = model_from_json(json.load(fin))
        else:
            self.model = Prophet(changepoint_prior_scale=0.5, interval_width = 0.95).fit(self.train_data)

    def read_csv_data(self, path):
        pdata = pd.read_csv(path, index_col= None, header=0)
        self.data = pdata[:1440]
        self.data.columns = ['ds', 'y']
        self.real_date = self.data.ds.values[-100:]
        test = pd.date_range("1995/07/01", periods=500, freq="D").values
        self.data['ds'] =test
    def split_train_test(self, train_rate =0.8):
        length_split = int(len(self.data)* train_rate)
        self.train_data = self.data[:length_split]
        self.test_data = self.data[length_split:].copy()

    def predict(self, t):
        return self.model.predict(t)
    def evaluate_by_metrics(self):
        self.build_model()
        x_test = pd.DataFrame(data = {'ds':self.test_data['ds']})
        forecast = self.predict(x_test)
        y_pred = forecast['yhat_upper']
        y_true = self.test_data['y']

        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        print('MAE: %.3f' % mae)
        #print('R2: %.3f' % r2)
        print('RMSE: %.3f' % rmse)
        # plot expected vs actual
        #print(y_pred)
        #print(y_true)

        self.test_data['y_pred'] = y_pred.values
        #print(self.test_data)
        dpath = "estimator/ts{}_test_{}.csv".format(self.tdata, self.tmodel)
        self.test_data['ds'] = self.real_date
        self.test_data.to_csv(dpath,index=None)
        #print(dpath)

        self.model.plot(forecast)
        pyplot.figure(figsize=(24, 6))
        pyplot.plot(x_test['ds'],y_true, label='Actual')
        pyplot.plot(x_test['ds'],y_pred, label='Predicted')
        pyplot.legend()
        pyplot.show()
# x = WorkloadEstimator()
# x.read_csv_data('estimator\data_est\clarknet_train_test.csv')
x = WorkloadEstimator(2)
x.read_csv_data('estimator\data_est/nasa_train_test.csv')
x.split_train_test(0.8)
x.evaluate_by_metrics()
#x.build_model()
#f = WorkloadEstimator().process_data(x, 5)