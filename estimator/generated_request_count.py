import numpy as np
import pandas as pd
from datetime import datetime,timedelta

from pyparsing import col
def task_type_clarknet():
    df = pd.read_csv("estimator\clarknet_dataset.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    start_time_first = pd.to_datetime('1995-08-28 00:00:00')
    end_time = start_time_first
    with open("estimator\clarknet_data_request_count_5min.csv","w") as output:
        number_task = len(df)
        while(number_task > 0):
            start_time = end_time
            end_time = start_time +timedelta(minutes=5)
            tdf = df[(df["datetime"] >= start_time) & (df["datetime"] < end_time)].reset_index()
            number_task = len(tdf)
            output.write("{},{}\n".format(start_time,number_task))

def filter_task_type_clarknet():
    df = pd.read_csv("estimator\clarknet_data_request_count_5min.csv", names =["datetime", "request_count"])

    df["datetime"] = pd.to_datetime(df["datetime"])
    ex_df = df[(df["datetime"].dt.hour*60 >= 8*60) & (df["datetime"].dt.hour *60 + df["datetime"].dt.minute < 16*60+20)].reset_index()
    #ex_df = df[(df["datetime"].dt.hour >= 8) & (df["datetime"].dt.hour <= 4 & df["datetime"].dt.minute < 20)].reset_index()
    ex_df.to_csv("estimator\clarknet_data_request_count_filter.csv", index=None)

def task_type_nasa():
    df = pd.read_csv("estimator/nasa_dataset.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    start_time_first = pd.to_datetime('1995-07-01 00:00:00')
    end_time = start_time_first
    with open("estimator/nasa_data_request_count_5min.csv","w") as output:
        number_task = len(df)
        while(number_task > 0):
            start_time = end_time
            end_time = start_time +timedelta(minutes=5)
            tdf = df[(df["datetime"] >= start_time) & (df["datetime"] < end_time)].reset_index()
            number_task = len(tdf)
            output.write("{},{}\n".format(start_time,number_task))

def filter_task_type_nasa():
    df = pd.read_csv("estimator/nasa_data_request_count_5min.csv", names =["datetime", "request_count"])

    df["datetime"] = pd.to_datetime(df["datetime"])
    ex_df = df[(df["datetime"].dt.hour*60 >= 8*60) & (df["datetime"].dt.hour *60 + df["datetime"].dt.minute < 16*60+20)].reset_index()
    #ex_df = df[(df["datetime"].dt.hour >= 8) & (df["datetime"].dt.hour <= 4 & df["datetime"].dt.minute < 20)].reset_index()
    ex_df.to_csv("estimator/nasa_data_request_count_filter.csv", index=None)

filter_task_type_nasa()