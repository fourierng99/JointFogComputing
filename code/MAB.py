import numpy as np
import matplotlib.pyplot as plt
import sys as sys
from environment import *
class Bandit:
    def __init__(self, k=10, exp_rate=.3, lr=0.1, ucb=False, seed=None, c=2,train=0,dataindex=0,number_server=4, is_autoscale = 0, ls_rsc = 3, tmodel = 'arima'):
        self.k = int(number_server)
        self.actions = range(self.k)
        self.exp_rate = exp_rate
        self.lr = lr
        self.end = False
        self.total_reward = 0
        self.avg_reward = []
        self.count_5minus = 0
        self.reward_5minus = 0


        if ucb:
            self.enviroment = VehicleEnv("UCB",train,dataindex,number_server,is_autoscale, ls_rsc, tmodel)
            self.mab = open("testUCB.csv","w")
            self.his_files = open("UCB_5phut.csv","w")
        else:
            self.enviroment = VehicleEnv("MAB",train,dataindex,number_server,is_autoscale, ls_rsc, tmodel)
            print(self.enviroment.train)
            self.mab = open("testMAB.csv","w")
            path_file = "result/MAB_{2}/MAB_5minute_s{0}_vm{1}_ts{2}.csv".format(self.k, self.enviroment.ls_rsc,self.enviroment.tdata)
            if(is_autoscale == 1):
                path_file = "result/MAB_{1}/MAB_5minute_s{0}_ts{1}_{2}.csv".format(self.k,self.enviroment.tdata, tmodel)
            print(path_file)
            self.his_files = open(path_file,"w")    
        self.his_files.write("count,reward,mean_reward\n")

        self.TrueValue = []
        
        np.random.seed(seed)
        #for i in range(self.k):
        #    self.TrueValue.append(np.random.randn() + 2)  # normal distribution

        self.values = np.zeros(self.k)
        self.times = 0
        self.action_times = np.zeros(self.k)

        self.ucb = ucb  # if select action using upper-confidence-bound
        self.c = c
        self.count_loop = 0
    def chooseAction(self):
        # explore
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # exploit
            if self.ucb:
                if self.times == 0:
                    action = np.random.choice(self.actions)
                else:
                    confidence_bound = self.values + self.c * np.sqrt(
                        np.log(self.times) / (self.action_times + 0.1))  # c=2
                    action = np.argmax(confidence_bound)
            else:
                action = np.argmax(self.values)
        return action

    def takeAction(self, action):
        self.count_5minus = self.count_5minus + 1
        self.times += 1
        self.action_times[action] += 1
        # take action and update value estimates
        # reward = self.TrueValue[action]
        m = self.enviroment.step(action)
        reward = m[1]
        self.reward_5minus = self.reward_5minus + reward
        done = m[2]
        if done:
            self.his_files.write("{},{},{}\n".format(self.count_5minus, self.reward_5minus, self.reward_5minus/self.count_5minus))
            self.count_5minus = 0
            self.count_loop += 1

            self.reward_5minus = 0
            self.end = True
            if self.count_loop != 100:
                self.enviroment.reset()
        # using incremental method to propagate
        self.values[action] += self.lr * (reward - self.values[action])  # look like fixed lr converges better

        self.total_reward += reward
        self.avg_reward.append(self.total_reward / self.times)
        self.mab.write(str(self.total_reward / self.times)+"\n")
    def play(self):
        for i in range(100):
            self.end = False 
            while not self.end:
                action = self.chooseAction()
                self.takeAction(action)


if __name__ == "__main__":


    bdt = Bandit(k=1, exp_rate=0.1, seed=1234, train=int(sys.argv[1]),dataindex=sys.argv[2],number_server=sys.argv[3],is_autoscale = int(sys.argv[4]),ls_rsc = int(sys.argv[5]), tmodel = sys.argv[6])
    bdt.play()
    avg_reward1 = bdt.avg_reward
    #bdt = Bandit(k=4, exp_rate=0.1, seed=1234, ucb=True, c=2)
    #bdt.play()
    #avg_reward3 = bdt.avg_reward
    plt.figure(figsize=[8, 6])
    plt.plot(avg_reward1, label="exp_rate=0.1")
   # plt.plot(avg_reward3, label="ucb, c=2")
    plt.xlabel("n_iter", fontsize=14)
    plt.ylabel("avg reward", fontsize=14)
    plt.legend()
    plt.show()