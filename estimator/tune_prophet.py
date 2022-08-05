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
from sklearn.metrics import mean_absolute_percentage_error
class TestWorkloadEstimator:
    def __init__(self,rate = 0.8,tdata =1):
        self.rate = 0.8
        self.tdata = tdata
        self.tmodel = 'prophet'
        self.path = 'estimator\data_est/clarknet_train_test.csv'
        if self.tdata ==2:
            self.path = 'estimator\data_est/nasa_train_test.csv'
        self.read_csv_data(self.path)
        self.split_train_test(self.rate)

    def build_model(self, scale = 0.5, n_point = 10, int_width = 0.95):
            self.model = Prophet(changepoint_prior_scale=scale, n_changepoints=n_point, interval_width = int_width, ).fit(self.train_data)
            #self.model = Prophet(changepoint_prior_scale=0.5, n_changepoints=200, interval_width = 0.95).fit(self.train_data)
    
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
        growth = ['linear', 'logistic']
        scales = [0.01,0.05,0.1]
        n_point = [5,10,20,50]

        self.tune_file = open('estimator/tune_prophet_{}.csv'.format(self.tdata), 'w')
        self.tune_file.write('scale,n_point,rmse,mape,mae\n')

        x_test = pd.DataFrame(data = {'ds':self.test_data['ds']})
        y_true = self.test_data['y']
        for g in growth:
            for i in scales:
                for j in n_point:
                    self.build_model(i, j, 0.95)
                    forecast = self.predict(x_test)
                    y_pred = forecast['yhat']

                    mae = mean_absolute_error(y_true, y_pred)
                    mape = mean_absolute_percentage_error(y_true, y_pred)
                    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

                    print('scale = {}, changepoint = {}, rmse = {}, mape = {}, mae = {}'.format(i,j,rmse,mape,mae))
                    self.tune_file.write('{},{},{},{},{}\n'.format(i,j,rmse,mape,mae))

x = TestWorkloadEstimator(tdata=1)
x.evaluate_by_metrics()


