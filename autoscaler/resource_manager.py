from bisect import bisect_right
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pkg_resources import working_set

class ResourceManager:
    def __init__(self, number_of_vm = 3,number_of_server = 4, ts_model = 'arima',ts_data = 1) :
        self.number_of_vm = number_of_vm
        self.ts_model = ts_model
        self.ts_data = ts_data
        self.number_of_server = number_of_server

        self.train_eps_data_path ="data/resource_manager_data/ts{}_train.csv".format(self.ts_data)
        self.test_eps_data_path = "data/resource_manager_data/ts{0}_test_{1}.csv".format(self.ts_data, self.ts_model) 

        self.train_eps_data = pd.read_csv(self.train_eps_data_path)
        self.test_eps_data = pd.read_csv(self.test_eps_data_path)
        self.data = {}

        for i in range(0, number_of_vm):
            val_div = 400
            train_path = 'autoscaler\data\DQL_5_minute_s{0}_vm{1}_ts{2}.csv'.format(self.number_of_server,i+1, self.ts_data)
            print(train_path)
            df = pd.read_csv(train_path)
            for j in range(len(df)):
                key = float(int(self.train_eps_data.loc[j].y/val_div)*val_div)
                val =df.loc[j].total_reward
                if( key in self.data.keys()):
                    self.data[key][i][0] = (self.data[key][i][0] + val)/(self.data[key][i][1]+1)
                    self.data[key][i][1] = self.data[key][i][1]+1
                else:
                    self.data[key] = [[0,0] for i in range(number_of_vm)]
                    self.data[key][i][0] =self.data[key][i][0] + val
                    self.data[key][i][1] = self.data[key][i][1]+1
       
    def update_scale_list(self, lst_num_request):
        lst_scale = []
        for e in lst_num_request:
            try:
                min_val = min(filter(lambda i : i>=e, self.data.keys()))
            except:
                min_val = max(self.data.keys())
            lst_scale.append(1+ [item[0] for item in self.data[min_val]].index(max([item[0] for item in self.data[min_val]])))
        return lst_scale
    
    def update_scale_list2(self, lst_num_request):
        lst_data = [0, 800, 1100]
        lst_scale = []
        for e in lst_num_request:
            if e < lst_data[1]:
                lst_scale.append(1)
            elif e < lst_data[2]:
                lst_scale.append(2)
            else:
                lst_scale.append(3)
        return lst_scale

    def export_train_test_scaling(self):
        # train_scale = self.update_scale_list(self.train_eps_data.y.values)
        # test_scale = self.update_scale_list(self.test_eps_data.y_pred.values)

        train_scale = self.update_scale_list2(self.train_eps_data.y.values)
        test_scale = self.update_scale_list2(self.test_eps_data.y_pred.values)

        train_df = pd.DataFrame()
        train_df['eps'] = [i for i in range(100)]
        train_df["scale"] = train_scale

        test_df = pd.DataFrame()
        test_df['eps'] = [i for i in range(100)]
        test_df["scale"] = test_scale

        train_df.to_csv('autoscaler/scale{0}_train_s{1}.csv'.format(self.ts_data, self.number_of_server), index=False)
        train_df.to_csv('autoscaler/scale{0}_test_s{1}_{2}.csv'.format(self.ts_data, self.number_of_server, self.ts_model),index=False)


        print(train_df.scale.values)
        print(test_df.scale.values)

x = ResourceManager(3,4,'arima',1)
#x = ResourceManager(3,4,'prophet',1)
x.export_train_test_scaling()