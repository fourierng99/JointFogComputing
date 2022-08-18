import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding
import copy
import os

from  config import *
from server import *
from auto_scaler import *
class VehicleEnv(gym.Env):
    def __init__(self,env,train,data,number_server,is_autoscale = 0, ls_rsc = 3.0, tmodel = 'prophet') :

        #Type of data : type 1 or 2
        self.tdata = data
        self.train = train
        self.vm_rsc = 1.5
        self.delta_vm = 100
        self.number_server = number_server
        self.env = env
        self.guess_count = 0
        self.number = 1
        self.ts_model = tmodel
        self.tasks_in_node = MAX_SERVER*[0]
        self.action_space = spaces.Discrete(MAX_SERVER)
        self.observation_space = spaces.Box(LOWER_OBSERVATON, HIGHER_OBSERVATION, [16])
        self.is_autoscale = is_autoscale
        self.auto_scaler = ResourceAutoScaler(self.ts_model,self.tdata, self.number_server)
        self.ls_rsc = ls_rsc*self.vm_rsc

        self.num_vms = ls_rsc

        self.normalize = 0.01
        self.coeff = 0.75
        # if(is_autoscale == 1):
        #     self.resource_autoscaler = ResourceManager(self, 3,'Prophert' ,self.tdata)
        # iniy list of avaiable server in env
        #1,1.2,1
        lst_server = []
        lst_server.append(VehicleServer(1, os.path.join(DATA_DIR, "data9000.xlsx")))
        lst_server.append(VehicleServer(1.2, os.path.join(DATA_DIR, "data9001.xlsx")))
        lst_server.append(VehicleServer(1, os.path.join(DATA_DIR , "data9002.xlsx")))

        self.server_pool = {"local": LocalServer(self.ls_rsc), "bus":lst_server}

        #streaming data of task
        if env != "DQN": 
            self.index_of_episode = 0
            if self.train == 0:
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

            self.observation = np.array([self.server_pool["bus"][0].get_vehicle_location(self.queue[0][0]),self.server_pool["bus"][0].qtime,self.server_pool["bus"][0].rsc\
                ,self.server_pool["bus"][1].get_vehicle_location(self.queue[0][0]),self.server_pool["bus"][1].qtime,self.server_pool["bus"][1].rsc\
                ,self.server_pool["bus"][2].get_vehicle_location(self.queue[0][0]),self.server_pool["bus"][2].qtime,self.server_pool["bus"][2].rsc\
                ,self.server_pool["local"].qtime,self.server_pool["local"].rsc,\
                self.queue[0][1],self.queue[0][2],self.queue[0][4]])
            
        else:
            self.index_of_episode = -1
            self.observation = np.array([-1])
            
            if(self.is_autoscale == 1):
                if(self.train == 0):
                    self.change_resource_local(self.auto_scaler.get_test_eps_data(0)*self.vm_rsc)
                else:
                    self.change_resource_local(self.auto_scaler.get_train_eps_data(0)*self.vm_rsc)
                #print(self.server_pool["local"].rsc, self.index_of_episode)

        #save result into file cs
                #configuration for connection radio between bus and 
        if (len(self.observation)>3):
            self.observation[-2] = self.observation[-2]/1000
            self.observation[-3] = self.observation[-3]/1000

        self.Pr = Config.Pr
        self.Pr2 = Config.Pr2
        self.Wm = Config.Wm
        self.o2 = 100

        self.set_result_file()
        self.sumreward = 0
        self.nreward = 0
        self.configuration_result_file.write("server,bus1,bus2,bus3\n")
        self.quality_result_file.write("good,medium,bad\n")

        self.seed()

        self.num_request_path = 'data/resource_manager_data/ts1_test_arima.csv'
        self.request_data  = pd.read_csv(self.num_request_path)
    
    def set_result_file(self):
        dir = os.path.join(RESULT_DIR,"{0}_{1}".format(self.env, self.tdata))
        if not os.path.exists(dir):
            os.makedirs(dir)
        if self.is_autoscale ==1 or self.env == 'MAB':
            self.rewardfiles = open("result/{0}_{1}/{0}_reward_s{2}.csv".format(self.env,self.tdata,self.number_server),"w")
            self.quality_result_file = open("result/{0}_{1}/{0}_n_quality_tasks_s{2}.csv".format(self.env,self.tdata,self.number_server),"w")
            self.configuration_result_file = open("result/{0}_{1}/{0}_config_parameter_s{2}.csv".format(self.env,self.tdata,self.number_server),"w")
            self.node_computing = open("result/{0}_{1}/{0}_task_offloading_s{2}.csv".format(self.env,self.tdata,self.number_server),"w")
            self.node_computing.write("number_of_server,distance,server_0,server_1,server_2,server_3,reward\n")
        else:
            reward_path = "result/{0}_{1}/ts{1}/{0}_reward_s{2}_vm{3}.csv".format(self.env,self.tdata,self.number_server, self.ls_rsc)
            quality_path ="result/{0}_{1}/ts{1}/{0}_n_quality_tasks_s{2}_vm{3}.csv".format(self.env,self.tdata,self.number_server, self.ls_rsc)
            configuration_path = "result/{0}_{1}/ts{1}/{0}_config_parameter_s{2}_vm{3}.csv".format(self.env,self.tdata,self.number_server, self.ls_rsc)
            node_computing_path = "result/{0}_{1}/ts{1}/{0}_task_offloading_s{2}_vm{3}.csv".format(self.env,self.tdata,self.number_server, self.ls_rsc)
            print(reward_path, quality_path, configuration_path, node_computing_path)

            self.rewardfiles = open(reward_path,"w")
            self.quality_result_file = open(quality_path,"w")
            self.configuration_result_file = open(configuration_path,"w")
            self.node_computing = open(node_computing_path,"w")
            self.node_computing.write("number_of_server,distance,server_0,server_1,server_2,server_3,reward\n")

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
        # if(self.env == 'DQN'):
                #     self.coeff = 0.7
                #     self.normalize = 0.01
                #     time_run = max(0,min((2*self.observation[13]-time_delay)/self.observation[13],1))
                #     energy = (3-self.server_pool["local"].rsc/self.vm_rsc)/2
                #     reward = self.coeff *time_run +(1-self.coeff)*energy + self.normalize
                #     # reward = min(1,self.coeff *time_run +(1-self.coeff)*energy + self.normalize)
        if(self.env == 'MAB'):
             self.coeff = 0.7

        self.tasks_in_node[action] = self.tasks_in_node[action]+1
        #reward = max(0,min((2*self.observation[13]-time_delay)/self.observation[13],1)) - self.delta_vm*int(self.server_pool["local"].rsc/self.vm_rsc)/self.request_data.y_pred.values[self.index_of_episode]
        time_run = max(0,min((2*self.observation[13]-time_delay)/self.observation[13],1))
        energy = ((3-self.server_pool["local"].rsc/self.vm_rsc)+ self.normalize)/2
        # if energy >1.0:
        #     energy = 1.0
        #reward = time_run*self.coeff
        reward = min(1,self.coeff *time_run +(1-self.coeff)*energy)

        self.node_computing.write(",{}\n".format(reward))
        if reward == 1 or time_run==1:
            self.n_quality_tasks[0]+=1
        elif reward ==0 or time_run ==0:
            self.n_quality_tasks[2] += 1
        else:
            self.n_quality_tasks[1] += 1
        
        if len(self.queue) != 0:
            self.queue = np.delete(self.queue,(0),axis=0)
        
        #check length of queue at this time and update state
        if len(self.queue) == 0 and len(self.data) != 0:
            self.queue = copy.deepcopy(self.data[self.data[:,0]==self.data[0][0]])
            
            for a in range(3):
                #self.observation[0+a*3] = self.readexcel(900+a,self.data[0][0])
                self.observation[0+a*3] = self.server_pool["bus"][action -1].get_vehicle_location(self.data[0][0])
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
            print(self.tasks_in_node)
            print(self.server_pool["local"].rsc, self.server_pool["bus"][0].rsc,self.server_pool["bus"][1].rsc,self.server_pool["bus"][2].rsc )
            self.configuration_result_file.write(str(self.tasks_in_node[0])+","+str(self.tasks_in_node[1])+","+str(self.tasks_in_node[2])+","+str(self.tasks_in_node[3])+","+"\n")
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

    def reset(self):
        #autoscaling the resources
        if(self.is_autoscale == 1):
            if(self.train == 1):
                print("here train" + str(self.auto_scaler.get_train_eps_data((self.index_of_episode +1))*self.vm_rsc))
                self.change_resource_local(self.auto_scaler.get_train_eps_data((self.index_of_episode +1))*self.vm_rsc)
            else:
                print("here test" + str(self.auto_scaler.get_train_eps_data((self.index_of_episode +1))*self.vm_rsc))
                self.change_resource_local(self.auto_scaler.get_test_eps_data((self.index_of_episode +1))*self.vm_rsc)
            #print(self.server_pool["local"].rsc, self.index_of_episode)

        if self.index_of_episode == -1: 
            self.index_of_episode = 0
            if self.train ==0:
                self.data = pd.read_csv(os.path.join(DATA_TASK,"data_{}_test/datatask{}.csv".format(self.tdata,self.index_of_episode)),header=None).to_numpy()
            else:
                self.data = pd.read_csv(os.path.join(DATA_TASK,"data_{}_train/datatask{}.csv".format(self.tdata,self.index_of_episode)),header=None).to_numpy()        
            self.data = np.sort(self.data, axis=0)
            #self.data[:,2] = self.data[:,2] / 1000.0
            #self.data[:,1] = self.data[:,1] / 1024.0
            

            self.n_quality_tasks = [0,0,0]
            self.queue = copy.deepcopy(self.data[self.data[:,0]==self.data[0][0]])
            self.data = self.data[self.data[:,0]!=self.data[0][0]]
            self.result = []
            self.time_last = self.data[-1][0]
            self.time = self.queue[0][0]

            #first observation of agent about eviroment
            self.observation = np.array([self.server_pool["bus"][0].get_vehicle_location(self.queue[0][0]),self.server_pool["bus"][0].qtime,self.server_pool["bus"][0].rsc\
                ,self.server_pool["bus"][1].get_vehicle_location(self.queue[0][0]),self.server_pool["bus"][1].qtime,self.server_pool["bus"][1].rsc\
                ,self.server_pool["bus"][2].get_vehicle_location(self.queue[0][0]),self.server_pool["bus"][2].qtime,self.server_pool["bus"][2].rsc\
                ,self.server_pool["local"].qtime,self.server_pool["local"].rsc,\
                self.queue[0][1],self.queue[0][2],self.queue[0][4]])

            self.observation[-2] = self.observation[-2]/1000
            self.observation[-3] = self.observation[-3]/1000
            return self.observation
        self.result = []
        self.number = 0
        self.guess_count = 0

        self.n_quality_tasks = [0, 0, 0]
        self.tasks_in_node=[0, 0, 0, 0]
        self.index_of_episode = self.index_of_episode + 1

        if self.index_of_episode>=100:
            self.index_of_episode = 0
        if self.train ==0:
            self.data = pd.read_csv(os.path.join(DATA_TASK,"data_{}_test/datatask{}.csv".format(self.tdata,self.index_of_episode)),header=None).to_numpy()
        else:
            self.data = pd.read_csv(os.path.join(DATA_TASK,"data_{}_train/datatask{}.csv".format(self.tdata,self.index_of_episode)),header=None).to_numpy()
        self.data = np.sort(self.data, axis=0)
        #self.data[:,2] = self.data[:,2] / 1000.0
        #self.data[:,1] = self.data[:,1] / 1024.0
        self.queue = copy.deepcopy(self.data[self.data[:,0]==self.data[0][0]])
        self.data = self.data[self.data[:,0]!=self.data[0][0]]
        self.time = self.queue[0][0]

        self.observation = np.array([self.server_pool["bus"][0].get_vehicle_location(self.queue[0][0]),\
            max(0,self.observation[1]-(self.time-self.time_last)),
            self.server_pool["bus"][0].rsc,\
            self.server_pool["bus"][1].get_vehicle_location(self.queue[0][0]), 
            max(0,self.observation[4]-(self.time-self.time_last)), 
            self.server_pool["bus"][1].rsc,\
            self.server_pool["bus"][2].get_vehicle_location(self.queue[0][0]), 
            max(0,self.observation[7]-(self.time-self.time_last)), 
            self.server_pool["bus"][2].rsc,\
            max(0,self.observation[9]-(self.time-self.time_last)), 
            self.server_pool["local"].rsc,\
            self.queue[0][1],self.queue[0][2], 
            self.queue[0][4]])
        try:
            self.time_last = self.data[-1][0]
        except:
            print("no self.timelast in esp{}".format(self.index_of_episode))
        self.observation[-2] = self.observation[-2]/1000
        self.observation[-3] = self.observation[-3]/1000
        return self.observation
    
    def change_resource_local(self, val):
        self.server_pool["local"].change_resource(val)
    
    def render(self,mode='human'):
        pass        