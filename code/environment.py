import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding
import copy
import os

from  config import *
from server import *
class VehicleEnv(gym.Env):
    def __init__(self,env,train,data,number_server) :

        #Type of data : type 1 or 2
        self.tdata = data
        self.train = train

        self.number_server = number_server
        self.env = env
        self.number = 1

        self.tasks_in_node = MAX_SERVER*[0]
        self.action_space = spaces.Discrete(MAX_SERVER)
        self.observation_space = spaces.Box(LOWER_OBSERVATON, HIGHER_OBSERVATION, [16])

        # iniy list of avaiable server in env
        
        lst_server = []
        lst_server.append(VehicleServer(1, os.path.join(DATA_DIR, "data9000.xlsx")))
        lst_server.append(VehicleServer(1.2, os.path.join(DATA_DIR, "data9001.xlsx")))
        lst_server.append(VehicleServer(1, os.path.join(DATA_DIR , "data9002.xlsx")))

        self.server_pool = {"local": LocalServer(3), "bus":lst_server}
        #streaming data of localtion of three bus with(900, 901, 902)
        # data900 = pd.read_excel(os.path.join(DATA_DIR, "data9000.xlsx"), index_col=0).to_numpy()
        # data900 = data900[:, 13:15]
        # data901 = pd.read_excel(os.path.join(DATA_DIR, "data9001.xlsx"), index_col=0).to_numpy()
        # data901 = data901[:, 13:15]
        # data902 = pd.read_excel(os.path.join(DATA_DIR , "data9002.xlsx"), index_col=0).to_numpy()
        # data902 = data902[:, 13:15]
        # self.data_bus = {"900":data900, "901":data901, "902":data902}

        #streaming data of task
        if env != "DQN": 
            self.index_of_episode = 0
            if self.train ==0:
                self.data = pd.read_csv(os.path.join(DATA_TASK,"data_{}_test/datatask{}.csv".format(self.tdata,self.index_of_episode)),header=None).to_numpy()
            else:
                self.data = pd.read_csv(os.path.join(DATA_TASK,"data_{}_train/datatask{}.csv".format(self.tdata,self.index_of_episode)),header=None).to_numpy()       
            self.data = np.sort(self.data, axis=0)
            self.n_quality_tasks = [0,0,0]
            self.queue = copy.deepcopy(self.data[self.data[:,0]==self.data[0][0]])
            self.data = self.data[self.data[:,0]!=self.data[0][0]] 
            self.result = []
            self.time_last = self.data[-1][0]
            self.time = self.queue[0][0]

            #first observation of agent about eviroment
            # self.observation = np.array([self.get_vehicle_location(900,self.queue[0][0]),0.0,1\
            #     ,self.get_vehicle_location(901,self.queue[0][0]),0,1.2\
            #     ,self.get_vehicle_location(902,self.queue[0][0]),0,1,\
            #     0,3,\
            #     self.queue[0][1],self.queue[0][2],self.queue[0][4]])

            self.observation = np.array([self.server_pool["bus"][0].get_vehicle_location(self.queue[0][0]),self.server_pool["bus"][0].qtime,self.server_pool["bus"][0].rsc\
                ,self.server_pool["bus"][1].get_vehicle_location(self.queue[0][0]),self.server_pool["bus"][1].qtime,self.server_pool["bus"][1].rsc\
                ,self.server_pool["bus"][2].get_vehicle_location(self.queue[0][0]),self.server_pool["bus"][2].qtime,self.server_pool["bus"][2].rsc\
                ,self.server_pool["local"].qtime,self.server_pool["local"].rsc,\
                self.queue[0][1],self.queue[0][2],self.queue[0][4]])
            
        else:
            self.index_of_episode = -1
            self.observation = np.array([-1])
        #save result into file cs
                #configuration for connection radio between bus and 
        if (len(self.observation)>3):
            self.observation[-2] = self.observation[-2]/1000
            self.observation[-3] = self.observation[-3]/1000

        self.Pr = Config.Pr
        self.Pr2 = Config.Pr2
        self.Wm = Config.Wm
        self.o2 = 100
        # if env == "MAB":
        #     self.rewardfiles = open("result/MAB{}/MAB_5phut_env_s{}.csv".format(self.tdata,self.number_server),"w")
        #     self.quality_result_file = open("result/MAB{}/n_quality_tasks_mab_s{}.csv".format(self.tdata,self.number_server),"w")
        #     self.configuration_result_file = open("result/MAB{}/thongso_mab_s{}.csv".format(self.tdata,self.number_server),"w")
        #     self.node_computing = open("result/MAB{}/chiatask_mab_s{}.csv".format(self.tdata,self.number_server),"w")
        #     self.node_computing.write("somay,distance,may0,may1,may2,may3,reward\n")
        # elif env == "UCB": 
        #     self.rewardfiles = open("UCB_5phut_env.csv","w")
        #     self.quality_result_file = open("n_quality_tasks_ucb.csv","w")
        #     self.configuration_result_file = open(os.path.join(RESULT_DIR, "thongso_ucb.csv"),"w")
        #     self.node_computing = open("chiatask_ucb.csv","w")
        #     self.node_computing.write("somay,distance,may0,may1,may2,may3,reward\n")
        # elif env == "DQN":
        #     self.rewardfiles = open("result/DQN{}/DQN_5phut_env_s{}.csv".format(self.tdata,self.number_server),"w")
        #     self.quality_result_file = open("result/DQN{}/n_quality_tasks_DQN_s{}.csv".format(self.tdata,self.number_server),"w")
        #     self.configuration_result_file = open("result/DQN{}/thongso_DQN_s{}.csv".format(self.tdata,self.number_server),"w")
        #     self.node_computing = open("result/DQN{}/chiatask_DQN_s{}.csv".format(self.tdata,self.number_server),"w")
        #     self.node_computing.write("somay,distance,may0,may1,may2,may3,reward\n")
        self.set_result_file()
        self.sumreward = 0
        self.nreward = 0
        self.configuration_result_file.write("server,bus1,bus2,bus3\n")
        self.quality_result_file.write("good,medium,bad\n")

        self.seed()
    
    def set_result_file(self):
        dir = os.path.join(RESULT_DIR,"{0}_{1}".format(self.env, self.tdata))
        if not os.path.exists(dir):
            os.makedirs(dir)
    
        self.rewardfiles = open("result/{0}_{1}/{0}_reward_s{2}.csv".format(self.env,self.tdata,self.number_server),"w")
        self.quality_result_file = open("result/{0}_{1}/{0}_n_quality_tasks_s{2}.csv".format(self.env,self.tdata,self.number_server),"w")
        self.configuration_result_file = open("result/{0}_{1}/{0}_config_parameter_s{2}.csv".format(self.env,self.tdata,self.number_server),"w")
        self.node_computing = open("result/{0}_{1}/{0}_task_offloading_s{2}.csv".format(self.env,self.tdata,self.number_server),"w")
        self.node_computing.write("number_of_server,distance,server_0,server_1,server_2,server_3,reward\n")

    
    # def get_vehicle_location(self, number_bus, time):
    #     data = self.data_bus[str(number_bus)]

    #     after_time = data[data[:,1] >= time]
    #     pre_time = data[data[:,1]<=time]
    #     if len(after_time) == 0:
    #         return 1.8
    #     las = after_time[0]
    #     first = pre_time[-1]
    #     if las[1] != first[1]:
    #         distance = (las[0] * (las[1]-time) + first[0] * (-first[1]+time)) / (las[1]-first[1])
    #     else:
    #         distance = las[0] 
    #     return distance

    def step(self, action):
        time_delay = 0
        #print(self.observation)
        #logic block when computing node is bus node
        if action>0 and action<4:
            Rate_trans_req_data = (10*np.log2(1+46/(np.power(self.observation[(action-1)*3],4)*100))) / 8
            #print(Rate_trans_req_data)
            self.observation[1+(action-1)*3] =  self.observation[11]/(self.observation[2+(action-1)*3]) + max(self.observation[12]/(Rate_trans_req_data),self.observation[1+(action-1)*3])
            
            #print(self.observation[1+(action-1)*3])

            #distance_response = self.readexcel(900+action-1,self.observation[1+(action-1)*3]+self.time)
            distance_response = self.server_pool["bus"][action -1].get_vehicle_location(self.observation[1+(action-1)*3]+self.time)

            Rate_trans_res_data = (10*np.log2(1+46/(np.power(distance_response,4)*100)))/8
            time_delay = self.observation[1+(action-1)*3]+self.queue[0][3]/(Rate_trans_res_data*1000)
            self.node_computing.write("{},{},{},{},{},{}".format(action,self.observation[(action-1)*3],self.observation[9],self.observation[1],self.observation[4],self.observation[7]))
        
        #logic block when computing node is server
        if action == 0:
            self.observation[9] += self.observation[11]/(self.observation[10])
            #import pdb;pdb.set_trace()

            time_delay = self.observation[9]
            self.node_computing.write("{},{},{},{},{},{}".format(action,0,self.observation[9],self.observation[1],self.observation[4],self.observation[7]))
        
        self.n_tasks_in_node[action] = self.n_tasks_in_node[action]+1
        reward = max(0,min((2*self.observation[13]-time_delay)/self.observation[13],1))
        self.node_computing.write(",{}\n".format(reward))
        
        if reward == 1:
            self.n_quality_tasks[0]+=1
        elif reward == 0:
            self.n_quality_tasks[2] += 1
        else:
            self.n_quality_tasks[1] += 1
        
        if len(self.queue) != 0:
            self.queue = np.delete(self.queue,(0),axis=0)
        
        #check length of queue at this time and update state
        if len(self.queue) == 0 and len(self.data) != 0:
            self.queue = copy.deepcopy(self.data[self.data[:,0]==self.data[0][0]])
            
            for a in range(3):
                self.observation[0+a*3] = self.readexcel(900+a,self.data[0][0])
            time = self.data[0][0] - self.time
            self.observation[1] = max(0,self.observation[1]-time)
            self.observation[4] = max(0,self.observation[4]-time)
            self.observation[7] = max(0,self.observation[7]-time)
            self.observation[9] = max(0,self.observation[9]-time)
            self.time = self.data[0][0]
            self.data = self.data[self.data[:,0]!=self.data[0,0]]
        
        if len(self.queue)!=0:
            self.observation[11] = self.queue[0][1]
            self.observation[12] = self.queue[0][2]
            self.observation[13] = self.queue[0][4]
        
        #check end of episode?
        done = len(self.queue) == 0 and len(self.data) == 0
        if done:
            print(self.n_tasks_in_node)
            self.configuration_result_file.write(str(self.n_tasks_in_node[0])+","+str(self.n_tasks_in_node[1])+","+str(self.n_tasks_in_node[2])+","+str(self.n_tasks_in_node[3])+","+"\n")
            self.quality_result_file.write("{},{},{}\n".format(self.n_quality_tasks[0],self.n_quality_tasks[1],self.n_quality_tasks[2]))
            
            #check end of program? to close files 
            if self.index_of_episode == 99:
                self.quality_result_file.close()
                self.configuration_result_file.close()
                self.node_computing.close()
        self.sumreward = self.sumreward + reward
        self.nreward = self.nreward + 1
        avg_reward = self.sumreward/self.nreward
        self.rewardfiles.write(str(avg_reward)+"\n")
        self.observation[-2] = self.observation[-2]/1000
        self.observation[-3] = self.observation[-3]/1000
        return self.observation, reward, done,{"number": self.number, "guesses": self.guess_count}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
x = VehicleEnv("MAB",0, 1,5)
print(x.observation)