import pandas as pd
from datetime import timedelta
def calculate_nb_step(tdata,start_date, n_episode):
    dataset_path = "data\clarknet_dataset.csv"
    if tdata==2:
        dataset_path = "data/nasa_dataset.csv"

    df = pd.read_csv(dataset_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    start_time = pd.to_datetime(start_date)
    end_time = start_time + n_episode*timedelta(minutes=5)
    tdf = df[(df["datetime"] >= start_time) & (df["datetime"] < end_time)]
    return len(tdf)
l_train = 0
#l_test = 0
for i in range(0,100):
    df = pd.read_csv('data_task\data_1_train\datatask{}.csv'.format(i))
    #df1 = pd.read_csv('data_task\data_1_test\datatask{}.csv'.format(i))
    l_train += len(df)
    #l_test += len(df1)
print(l_train)