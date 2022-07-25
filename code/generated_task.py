from datetime import datetime, timedelta
import random as rd
import numpy as np
import pandas as pd
from  numpy.random import poisson as ps
from pathlib import Path
import os
import sys
path =os.path.abspath(__file__)
path =Path(path).parent.parent
def random_task_type_clarknet():
    df = pd.read_csv("data\clarknet_dataset.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    start_time_first = pd.to_datetime('1995-08-31 08:00:00')
    end_time = start_time_first
    for i in range(100):
        start_time = end_time
        end_time = start_time +timedelta(minutes=5)
        tdf = df[(df["datetime"] >= start_time) & (df["datetime"] < end_time)].reset_index()
        with open("{}/data_task/{}/datatask{}.csv".format(str(path),"data_1_train",i),"w") as output:
            number_task = len(tdf)
            for j in range(number_task):
                task = tdf.loc[j]
                m = (task.datetime-start_time_first).total_seconds()
                m1 = np.random.randint(1000,2000)
                m2 = np.random.randint(100,200)
                m3 = np.random.randint(500,1500)
                m4 = 1+np.random.rand()
                output.write("{},{},{},{},{}\n".format(m,m3,m1,m2,m4))
    
    start_time_first = pd.to_datetime('1995-09-01 08:00:00')
    end_time = start_time_first
    for i in range(100):
        start_time = end_time
        end_time = start_time +timedelta(minutes=5)
        tdf = df[(df["datetime"] >= start_time) & (df["datetime"] < end_time)].reset_index()
        with open("{}/data_task/{}/datatask{}.csv".format(str(path),"data_1_test",i),"w") as output:
            number_task = len(tdf)
            for j in range(number_task):
                task = tdf.loc[j]
                m = (task.datetime-start_time_first).total_seconds()
                m1 = np.random.randint(1000,2000)
                m2 = np.random.randint(100,200)
                m3 = np.random.randint(500,1500)
                m4 = 1+np.random.rand()
                output.write("{},{},{},{},{}\n".format(m,m3,m1,m2,m4))

def random_task_type_nasa():
    df = pd.read_csv("data/nasa_dataset.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    start_time_first = pd.to_datetime('1995-07-01 08:00:00')
    end_time = start_time_first
    for i in range(100):
        start_time = end_time
        end_time = start_time +timedelta(minutes=5)
        tdf = df[(df["datetime"] >= start_time) & (df["datetime"] < end_time)].reset_index()
        with open("{}/data_task/{}/datatask{}.csv".format(str(path),"data_2_train",i),"w") as output:
            number_task = len(tdf)
            for j in range(number_task):
                task = tdf.loc[j]
                m = (task.datetime-start_time_first).total_seconds()
                m1 = np.random.randint(1000,2000)
                m2 = np.random.randint(100,200)
                m3 = np.random.randint(500,1500)
                m4 = 1+np.random.rand()
                output.write("{},{},{},{},{}\n".format(m,m3,m1,m2,m4))
    
    start_time_first = pd.to_datetime('1995-07-05 08:00:00')
    end_time = start_time_first
    for i in range(100):
        start_time = end_time
        end_time = start_time +timedelta(minutes=5)
        tdf = df[(df["datetime"] >= start_time) & (df["datetime"] < end_time)].reset_index()
        with open("{}/data_task/{}/datatask{}.csv".format(str(path),"data_2_test",i),"w") as output:
            number_task = len(tdf)
            for j in range(number_task):
                task = tdf.loc[j]
                m = (task.datetime-start_time_first).total_seconds()
                m1 = np.random.randint(1000,2000)
                m2 = np.random.randint(100,200)
                m3 = np.random.randint(500,1500)
                m4 = 1+np.random.rand()
                output.write("{},{},{},{},{}\n".format(m,m3,m1,m2,m4))

# if __name__=="__main__":
#     types = 1
#     if len(sys.argv) > 1:
#         types = int (sys.argv[1])
#     if types ==1:
#         random_task_type_1(1100)
#     elif types==2:
#         random_task_type_2()
#     else:
#         print("vui lòng chọn kiểu dữ liệu type 1 or 2")
#random_task_type_clarknet()
random_task_type_nasa()