import numpy as np
from config import DATA_DIR, RESULT_DIR, Config
import pandas as pd
import os 
import glob
from prophet.serialize import model_to_json, model_from_json

class WorkloadEstimator:
    def __init__(self):
        pass

    def build_model(self, model_path = ''):
        if os.path.isfile(model_path):
            with open('serialized_model.json', 'r') as fin:
                self.model = model_from_json(json.load(fin))
        else:
            self.model = Prophet(changepoint_prior_scale=0.5, interval_width = 0.95).fit(self.train_data)

    def read_data(self):
        path =  os.path.join(DATA_DIR, "process_data")
        all_files = glob.glob(path + "/*.csv")
        li = []
        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)
        frame = pd.concat(li, axis=0, ignore_index=True)
        frame['date_time'] = pd.to_datetime(frame['date_time'])
        self.train_data = frame
        return frame

    def process_data(self, data, n = 5):
        d = {'date_time': 'first','request_count': 'sum'}
        return data.groupby(data.index // n).agg(d)

    def predict(self, t):
        return self.model.predict(t)
        
x = WorkloadEstimator().read_data()
f = WorkloadEstimator().process_data(x, 5)
print(f.head())
