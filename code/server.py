import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding
import copy
import os
from  config import *
class Server:
    def __init__(self,stype ,rsc):
        self.type = stype
        self.rsc = rsc
        self.qtime = 0.0
    
    def get_vehicle_location(self, time):
        return 0.0
    
    def get_observation(self, time):
        pass
    
    def reset(self):
        self.qtime = 0.0

class LocalServer(Server):
    def __init__(self, rsc=3):
        super().__init__("local", rsc)

    def get_observation(self, time):
        return np.array(self.qtime, self.rsc)

class VehicleServer(Server):
    def __init__(self,rsc =1, path = ""):
        super().__init__("bus", rsc)
        self.location_data = pd.read_excel(path, index_col=0).to_numpy()  
        self.location_data = self.location_data[:, 13:15]
    
    def get_vehicle_location(self, time):
        data = self.location_data

        after_time = data[data[:,1] >= time]
        pre_time = data[data[:,1]<=time]
        if len(after_time) == 0:
            return 1.8
        letzt = after_time[0]
        first = pre_time[-1]
        if letzt[1] != first[1]:
            distance = (letzt[0] * (letzt[1]-time) + first[0] * (-first[1]+time)) / (letzt[1]-first[1])
        else:
            distance = letzt[0] 
        return distance
    
    def get_observation(self,time):
        return np.array(self.get_vehicle_location(time),self.qtime, self.rsc)