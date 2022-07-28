import numpy as np
import pandas as pd
import os 
#from prophet.serialize import model_to_json, model_from_json
from prophet import Prophet

from datetime import datetime
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error
import math 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

class WorkloadEstimator:
    def __init__(self, tmodel, tdata):
        self.tmodel = tmodel
        self.tdata = tdata
        self.train_path = 'data/resource_manager_data/ts{0}_train.csv'.format(self.tdata)
        self.test_path = 'data/resource_manager_data/ts{0}_test_{1}.csv'.format(self.tdata,self.tmodel)
        
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)
        
    def get_train_data(self):
        return self.train_data
        
    def get_predict_data(self):
        return self.test_data

x = WorkloadEstimator('prophet',1)
x.get_train_data()
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
