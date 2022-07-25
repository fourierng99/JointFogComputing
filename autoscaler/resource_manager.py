from bisect import bisect_right
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pkg_resources import working_set
from workload_estimator import *

class ResourceManager:
    def __init__(self, number_of_vm, ts_model,ts_data) :
        self.number_of_vm = number_of_vm
        self.data = {}
        for i in range(0, number_of_vm):
            df = pd.read_csv('autoscaler\data\\reward_{}vm.csv'.format(i+1))
            for j in range(len(df)):
                q_data = df.loc[j]
                key = q_data.request_count
                val =q_data.total_reward
                if( key in self.data.keys()):
                    self.data[key][i][0] = (self.data[key][i][0] + val)/(self.data[key][i][1]+1)
                    self.data[key][i][1] = self.data[key][i][1]+1
                else:
                    self.data[key] = [[0,0] for i in range(number_of_vm)]
                    self.data[key][i][0] =self.data[key][i][0] + val
                    self.data[key][i][1] = self.data[key][i][1]+1
        
        self.ts_model = WorkloadEstimator(ts_model,ts_data)
    def auto_scale(self, eps):
        try:
            if(eps== -1):
                return self.list_scale[0]
            return self.list_scale[eps]
        except:
            return 3
    def update_scale_list(self, lst_num_request):
        lst_scale = []
        for e in lst_num_request:
            try:
                min_val = min(filter(lambda i : i>=e, self.data.keys()))
            except:
                min_val = max(self.data.keys())
            lst_scale.append(1+ [item[0] for item in self.data[min_val]].index(max([item[0] for item in self.data[min_val]])))
        self.list_scale=lst_scale
    
    def predict_scaling(self, start_date, num_eps):
        #need complete here
        self.list_ts = self.ts_model.predict()
        self.update_scale_list(self.list_ts)