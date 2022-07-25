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
    def __init__(self, tmodel, tdata, train_rate = 0.8):
        self.train_rate = train_rate
        if tmodel == "Prophert":
            self.build_model()
        if(tdata == 1):
            self.data_path = ''
        else:
            self.data_path =''
        self.read_csv_data(self.data_path)
        self.split_train_test(self.train_rate)

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

    def split_train_test(self, train_rate =0.8):
        length_split = int(len(self.data)* train_rate)
        self.train_data = self.data[:length_split]
        self.test_data = self.data[length_split:]

    def predict(self, t):
        return self.model.predict(t)

    def evaluate_by_metrics(self):
        self.build_model()
        x_test = pd.DataFrame(data = {'ds':self.test_data['ds']})
        forecast = self.predict(x_test)
        y_pred = forecast['yhat']
        y_true = self.test_data['y']

        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        print('MAE: %.3f' % mae)
        print('R2: %.3f' % r2)
        print('RMSE: %.3f' % rmse)
        # plot expected vs actual
        print(forecast)
        print(y_true)
        pyplot.figure(figsize=(24, 6))
        pyplot.plot(x_test['ds'],y_true, label='Actual')
        pyplot.plot(x_test['ds'],y_pred, label='Predicted')
        pyplot.legend()
        pyplot.show()

x = WorkloadEstimator()
x.read_csv_data('estimator\clarknet_data_request.csv')
x.split_train_test(0.8)
x.evaluate_by_metrics()
#x.build_model()
#f = WorkloadEstimator().process_data(x, 5)


# import numpy as np
# from config import DATA_DIR, RESULT_DIR, Config
# import pandas as pd
# import os 
# import glob
# from prophet.serialize import model_to_json, model_from_json

# class WorkloadEstimator:
#     def __init__(self):
#         pass

#     def build_model(self, model_path = ''):
#         if os.path.isfile(model_path):
#             with open('serialized_model.json', 'r') as fin:
#                 self.model = model_from_json(json.load(fin))
#         else:
#             self.model = Prophet(changepoint_prior_scale=0.5, interval_width = 0.95).fit(self.train_data)

#     def read_data(self):
#         path =  os.path.join(DATA_DIR, "process_data")
#         all_files = glob.glob(path + "/*.csv")
#         li = []
#         for filename in all_files:
#             df = pd.read_csv(filename, index_col=None, header=0)
#             li.append(df)
#         frame = pd.concat(li, axis=0, ignore_index=True)
#         frame['date_time'] = pd.to_datetime(frame['date_time'])
#         self.train_data = frame
#         return frame

#     def process_data(self, data, n = 5):
#         d = {'date_time': 'first','request_count': 'sum'}
#         return data.groupby(data.index // n).agg(d)

#     def predict(self, t):
#         return self.model.predict(t)
        
# x = WorkloadEstimator().read_data()
# f = WorkloadEstimator().process_data(x, 5)
# print(f.head())
