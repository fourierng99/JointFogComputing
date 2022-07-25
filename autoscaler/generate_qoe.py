import pandas as pd
import numpy as np
def generate_qoe_episode(number_of_vm):

    lst_length = []
    for i in range(0,100):
        eps_df = pd.read_csv("data_task\data_3_train\datatask{}.csv".format(i))
        lst_length.append(len(eps_df))
    for j in range(1, number_of_vm+1):
            df = pd.read_csv('autoscaler\data\\reward_s{}.csv'.format(j))
            df['request_count'] = lst_length
            df.to_csv('autoscaler\data\\reward_{}vm.csv'.format(j), index= False)
generate_qoe_episode(3)