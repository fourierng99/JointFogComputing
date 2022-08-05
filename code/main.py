from tensorflow.keras.optimizers import Adam
import copy
import json
import timeit
import warnings
from tempfile import mkdtemp
import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from rl.agents.ddpg import DDPGAgent
from rl.agents.sarsa import SARSAAgent
from rl.callbacks import Callback, FileLogger, ModelIntervalCheckpoint
from rl.memory import SequentialMemory
#from rl.policy import EpsGreedyQPolicy
from rl.random import OrnsteinUhlenbeckProcess
from tensorflow.keras.backend import cast
from tensorflow.keras.layers import (Activation, Concatenate, Dense, Dropout,
                                     Flatten, Input)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import sys
from datetime import datetime, timedelta
from environment import *
from policy import CustomerEpsGreedyQPolicy
from callback import *
import os

from rl.agents.dqn import DQNAgent

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
def build_model(state_size, num_actions):
    input = Input(shape=(1,state_size))
    x = Flatten()(input)
    x = Dense(16, activation='relu')(x)

    x = Dense(32, activation='relu')(x)

    x = Dense(32, activation='relu')(x)
  
    x = Dense(8, activation='relu')(x)

    output = Dense(num_actions, activation='linear')(x)
    model = Model(inputs=input, outputs=output)
    return model

def Run_DQN(env,number_server):
    model=build_model(14,number_server)
    num_actions = number_server
    policy = CustomerEpsGreedyQPolicy(0.1,0.1)
    env.seed(123)
    memory = SequentialMemory(limit=5000, window_length=1)
    
    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,\
              target_model_update=1e-3, policy=policy,gamma=0.9,memory_interval=1)
    
    callback_path = "DQL_5_minute_s{}_ts{}_{}.csv".format(env.number_server, env.tdata, env.ts_model)
    if(env.is_autoscale == 0):
        callback_path = "DQL_5_minute_s{}_vm{}_ts{}.csv".format(env.number_server,env.ls_rsc,env.tdata)

    callbacks = CustomerTrainEpisodeLogger(env = env ,filename= callback_path)
    callback2 = ModelIntervalCheckpoint("weight_DQLs{}.h5".format(number_server),interval=50000)
    dqn.compile(Adam(lr=1e-3), metrics=['mse','mae'])

    start_date = "1995-08-31 08:00:00"
    if(env.tdata == 2):
        start_date = "1995-07-04 8:00:00"
    print(callback_path, start_date)
    step = calculate_nb_step(env.tdata, start_date, 100) 

    dqn.fit(env, nb_steps= step , visualize=False, verbose=2,callbacks=[callbacks,callback2])
    #dqn.fit(env, nb_steps= 104774, visualize=False, verbose=2,callbacks=[callbacks,callback2])
    weight_path = "model_weights/DQN_weights_ts{0}_s{1}_{2}_.h5".format(env.tdata, env.number_server, env.ts_model)
    if(env.is_autoscale == 0):
        weight_path = "model_weights/DQN_weights_ts{0}_s{1}_vm{2}_.h5".format(env.tdata, env.number_server, env.ls_rsc)
    dqn.save_weights(weight_path)


def Test_DQN(env,number_server):
    model=build_model(14,number_server)
    num_actions = number_server
    policy = CustomerEpsGreedyQPolicy(0.1)
    env.seed(123)
    memory = SequentialMemory(limit=5000, window_length=1)

    test_path = "result/DQN_{1}/test_5minute_env_s{0}_ts{1}_{2}.csv".format(number_server, env.tdata, env.ts_model)
    if(env.is_autoscale==0):
        test_path ="result/DQN_{1}/test_5minute_env_s{0}_ts{1}_vm{2}.csv".format(number_server, env.tdata, env.ls_rsc)
    
    test_callback = CustomerTestLogger(test_path)
    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,\
              target_model_update=1e-3, policy=policy,gamma=0.7,memory_interval=1)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    weight_path = "model_weights/DQN_weights_ts{0}_s{1}_{2}_.h5".format(env.tdata, env.number_server, env.ts_model)
    if(env.is_autoscale == 0):
        weight_path = "model_weights/DQN_weights_ts{0}_s{1}_vm{2}_.h5".format(env.tdata, env.number_server, env.ls_rsc)
    #weight_path = 'model_weights\DQN_weights_ts1_s4_vm3_.h5'
    print(weight_path)
    dqn.load_weights(weight_path)
    dqn.test(env,100,callbacks=[test_callback])

def calculate_nb_step(tdata,start_date, n_episode):
    
    if tdata==2:
        l_train = 0
        for i in range(0,100):
            df = pd.read_csv('data_task\data_2_train\datatask{}.csv'.format(i))
            l_train += len(df)
        return l_train
    else:
        l_train = 0
        for i in range(0,100):
            df = pd.read_csv('data_task\data_1_train\datatask{}.csv'.format(i))
            l_train += len(df)
        return l_train
    
    
    dataset_path = "data\clarknet_dataset.csv"

    df = pd.read_csv(dataset_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    start_time = pd.to_datetime(start_date)
    end_time = start_time + n_episode*timedelta(minutes=5)
    tdf = df[(df["datetime"] >= start_time) & (df["datetime"] < end_time)]
    return len(tdf)

if __name__=="__main__":
    if len(sys.argv) > 1:
        types = sys.argv[1]
        train = int(sys.argv[2])
        data = int(sys.argv[3])
        number_server = int(sys.argv[4])
        is_autoscale = int(sys.argv[5])
        ls_rsc = int(sys.argv[6])
        tmodel = sys.argv[7]
        env = VehicleEnv(types,train,data,number_server, is_autoscale, ls_rsc, tmodel)
    if train:
        if types == "DQN":
            Run_DQN(env,number_server)
        
    else:
        if types == "DQN":
            Test_DQN(env,number_server)      
