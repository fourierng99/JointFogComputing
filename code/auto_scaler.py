from  config import *
import pandas as pd
class ResourceAutoScaler:
    def __init__(self,tmodel= 'prophet', tdata =1, number_of_server =4):
        self.tmodel = tmodel
        self.tdata = tdata
        self.number_of_server = number_of_server

        self.train_path = 'data/resource_manager_data/auto_scale/scale{0}_train_s{1}.csv'.format(self.tdata,self.number_of_server)
        self.test_path = 'data/resource_manager_data/auto_scale/scale{0}_test_s{1}_{2}.csv'.format(self.tdata,self.number_of_server,self.tmodel)
        
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)
        #print(self.train_data)
        # print(self.test_data)

    def get_train_eps_data(self,eps):
        try:
            return self.train_data.loc[eps].scale
        except:
            return 3
        
    def get_test_eps_data(self,eps):
        try:
            return self.test_data.loc[eps].scale
        except:
            return 3
    # def calculate_servers(self,request_seq):
    #     x = max(request_seq)
    #     volume = x/float(self.max_server)
    #     vm_seq = []
    #     for r in request_seq:
    #         v = float(r)/volume
    #         if(v> round(v)):
    #             vm_seq.append(min(round(v+1),self.max_server ))
    #         else:
    #             vm_seq.append(min(round(v),self.max_server ))
        
    #     self.vms = vm_seq
    #     return vm_seq


#asl = ResourceAutoScaler('prophet',1,4)
#val = asl.train_data
#print(asl.get_train_eps_data(0))
#print(asl.vms)